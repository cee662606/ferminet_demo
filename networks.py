# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Fermionic Neural Network in JAX."""
import enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import jastrows
from ferminet import network_blocks
import jax
import jax.numpy as jnp
from jax.nn.initializers import (
    ones,
    lecun_normal
)
from typing_extensions import Protocol


FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[
    jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']
]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = MutableMapping[str, jnp.ndarray]


@chex.dataclass
class FermiNetData:
  """Data passed to network.

  Shapes given for an unbatched element (i.e. a single MCMC configuration).

  NOTE:
    the networks are written in batchless form. Typically one then maps
    (pmap+vmap) over every attribute of FermiNetData (nb this is required if
    using KFAC, as it assumes the FIM is estimated over a batch of data), but
    this is not strictly required. If some attributes are not mapped, then JAX
    simply broadcasts them to the mapped dimensions (i.e. those attributes are
    treated as identical for every MCMC configuration.

  Attributes:
    positions: walker positions, shape (nelectrons*ndim).
    spins: spins of each walker, shape (nelectrons).
    atoms: atomic positions, shape (natoms*ndim).
    charges: atomic charges, shape (natoms).
  """

  # We need to be able to construct instances of this with leaf nodes as jax
  # arrays (for the actual data) and as integers (to use with in_axes for
  # jax.vmap etc). We can type the struct to be either all arrays or all ints
  # using Generic, it just slightly complicates the type annotations in a few
  # functions (i.e. requires FermiNetData[jnp.ndarray] annotation).
  positions: Any
  spins: Any
  atoms: Any
  charges: Any


## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclei charges, shape: (natoms).
    """


class LogFermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclear charges, shape: (natoms).
    """


class OrbitalFnLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      Sequence of orbitals.
    """


class InitLayersFn(Protocol):

  def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
    """Returns output dim and initialized parameters for the interaction layers.

    Args:
      key: RNG state
    """


class ApplyLayersFn(Protocol):

  def __call__(
      self,
      params: ParamTree,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Forward evaluation of the equivariant interaction layers.

    Args:
      params: parameters for the interaction and permutation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """


## Interfaces (network components) ##


class FeatureInit(Protocol):

  def __call__(self) -> Tuple[Tuple[int, int], Param]:
    """Creates the learnable parameters for the feature input layer.

    Returns:
      Tuple of ((x, y), params), where x and y are the number of one-electron
      features per electron and number of two-electron features per pair of
      electrons respectively, and params is a (potentially empty) mapping of
      learnable parameters associated with the feature construction layer.
    """


class FeatureApply(Protocol):

  def __call__(
      self,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      **params: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Creates the features to pass into the network.

    Args:
      ae: electron-atom vectors. Shape: (nelectron, natom, 3).
      r_ae: electron-atom distances. Shape: (nelectron, natom, 1).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, 3).
      r_ee: electron-electron distances. Shape: (nelectron, nelectron).
      **params: learnable parameters, as initialised in the corresponding
        FeatureInit function.
    """


@attr.s(auto_attribs=True)
class FeatureLayer:
  init: FeatureInit
  apply: FeatureApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):

  def __call__(
      self,
      natoms: int,
      nspins: Sequence[int],
      ndim: int,
      **kwargs: Any,
  ) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      natoms: number of atoms.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      **kwargs: additional kwargs to use for creating the specific FeatureLayer.
    """


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
  """Options controlling the overall network architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope: Envelope object to create and apply the multiplicative envelope.
    feature_layer: Feature object to create and apply the input features for the
      one- and two-electron layers.
    jastrow: Type of Jastrow factor if used, or 'none' if no Jastrow factor.
    complex_output: If true, the network outputs complex numbers.
  """

  ndim: int = 3
  determinants: int = 16
  states: int = 0
  full_det: bool = True
  rescale_inputs: bool = False
  bias_orbitals: bool = False
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          envelopes.make_isotropic_envelope,
          takes_self=False))
  feature_layer: FeatureLayer = None
  jastrow: jastrows.JastrowType = jastrows.JastrowType.NONE
  complex_output: bool = False
  system: Any


@attr.s(auto_attribs=True, kw_only=True)
class FermiNetOptions(BaseNetworkOptions):
  """Options controlling the FermiNet architecture.

  Attributes:
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    separate_spin_channels: If True, use separate two-electron streams for
      spin-parallel and spin-antiparallel  pairs of electrons. If False, use the
      same stream for all pairs of electrons.
    schnet_electron_electron_convolutions: Tuple of embedding dimension to use
      for a SchNet-style convolution between the one- and two-electron streams
      at each layer of the network. If empty, the original FermiNet embedding is
      used.
    nuclear_embedding_dim: dimension of nuclear embedding to use for
      SchNet-style embeddings. If falsy, not used.
    electron_nuclear_aux_dims: Dimensions of each layer of the electron-nuclear
      auxiliary stream. If falsy, not used.
    schnet_electron_nuclear_convolutions: Dimension of the SchNet-style
      convolution between the nuclear embedding and the electron-nuclear
      auxiliary stream at each layer. If falsy, not used.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
  """

  hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32), (256, 32))
  separate_spin_channels: bool = False
  schnet_electron_electron_convolutions: Tuple[int, ...] = ()
  nuclear_embedding_dim: int = 0
  electron_nuclear_aux_dims: Tuple[int, ...] = ()
  schnet_electron_nuclear_convolutions: Tuple[int, ...] = ()
  use_last_layer: bool = False


# Network class.


@attr.s(auto_attribs=True)
class Network:
  options: BaseNetworkOptions
  init: InitFermiNet
  apply: FermiNetLike
  apply_sym: FermiNetLike
  apply_osym: FermiNetLike
  orbitals: OrbitalFnLike


# Internal utilities


def _split_spin_pairs(
    arr: jnp.ndarray,
    nspins: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Splits array into parallel and anti-parallel spin channels.

  For an array of dimensions (nelec, nelec, ...), where nelec = sum(nspins),
  and the first nspins[0] elements along the first two axes correspond to the up
  electrons, we have an array like:

    up,up   | up,down
    down,up | down,down

  Split this into the diagonal and off-diagonal blocks. As nspins[0] !=
  nspins[1] in general, flatten the leading two dimensions before combining the
  blocks.

  Args:
    arr: array with leading dimensions (nelec, nelec).
    nspins: number of electrons in each spin channel.

  Returns:
    parallel, antiparallel arrays, where
       - parallel is of shape (nspins[0]**2 + nspins[1]**2, ...) and the first
         nspins[0]**2 elements correspond to the up,up block and the subsequent
         elements to the down,down block.
       - antiparallel is of shape (2 * nspins[0] + nspins[1], ...) and the first
         nspins[0] + nspins[1] elements correspond to the up,down block and the
         subsequent
         elements to the down,up block.
  """
  if len(nspins) != 2:
    raise ValueError(
        'Separate spin channels has not been verified with spin sampling.'
    )
  up_up, up_down, down_up, down_down = network_blocks.split_into_blocks(
      arr, nspins
  )
  trailing_dims = jnp.shape(arr)[2:]
  parallel_spins = [
      up_up.reshape((-1,) + trailing_dims),
      down_down.reshape((-1,) + trailing_dims),
  ]
  antiparallel_spins = [
      up_down.reshape((-1,) + trailing_dims),
      down_up.reshape((-1,) + trailing_dims),
  ]
  return (
      jnp.concatenate(parallel_spins, axis=0),
      jnp.concatenate(antiparallel_spins, axis=0),
  )


def _combine_spin_pairs(
    parallel_spins: jnp.ndarray,
    antiparallel_spins: jnp.ndarray,
    nspins: Tuple[int, int],
) -> jnp.ndarray:
  """Combines arrays of parallel spins and antiparallel spins.

  This is the reverse of _split_spin_pairs.

  Args:
    parallel_spins: array of shape (nspins[0]**2 + nspins[1]**2, ...).
    antiparallel_spins: array of shape (2 * nspins[0] * nspins[1], ...).
    nspins: number of electrons in each spin channel.

  Returns:
    array of shape (nelec, nelec, ...).
  """
  if len(nspins) != 2:
    raise ValueError(
        'Separate spin channels has not been verified with spin sampling.'
    )
  nsame_pairs = [nspin**2 for nspin in nspins]
  same_pair_partitions = network_blocks.array_partitions(nsame_pairs)
  up_up, down_down = jnp.split(parallel_spins, same_pair_partitions, axis=0)
  up_down, down_up = jnp.split(antiparallel_spins, 2, axis=0)
  trailing_dims = jnp.shape(parallel_spins)[1:]
  up = jnp.concatenate(
      (
          up_up.reshape((nspins[0], nspins[0]) + trailing_dims),
          up_down.reshape((nspins[0], nspins[1]) + trailing_dims),
      ),
      axis=1,
  )
  down = jnp.concatenate(
      (
          down_up.reshape((nspins[1], nspins[0]) + trailing_dims),
          down_down.reshape((nspins[1], nspins[1]) + trailing_dims),
      ),
      axis=1,
  )
  return jnp.concatenate((up, down), axis=0)


## Network layers: features ##


def construct_input_features(
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron and atomic positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    ae, ee, r_ae, r_ee tuple, where:
      ae: atom-electron vector. Shape (nelectron, natom, ndim).
      ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  #print("atoms.shape",atoms.shape,atoms.shape[1])
  assert atoms.shape[1] == ndim
  ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as is has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
  return ae, ee, r_ae, r_ee[..., None]


def make_ferminet_features(
    natoms: int,
    nspins: Optional[Tuple[int, int]] = None,
    ndim: int = 3,
    rescale_inputs: bool = False,
) -> FeatureLayer:
  """Returns the init and apply functions for the standard features."""

  del nspins

  def init() -> Tuple[Tuple[int, int], Param]:
    return (natoms * (ndim + 1), ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if rescale_inputs:
      log_r_ae = jnp.log(1 + r_ae)  # grows as log(r) rather than r
      ae_features = jnp.concatenate((log_r_ae, ae * log_r_ae / r_ae), axis=2)

      log_r_ee = jnp.log(1 + r_ee)
      ee_features = jnp.concatenate((log_r_ee, ee * log_r_ee / r_ee), axis=2)

    else:
      ae_features = jnp.concatenate((r_ae, ae), axis=2)
      ee_features = jnp.concatenate((r_ee, ee), axis=2)
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    return ae_features, ee_features

  return FeatureLayer(init=init, apply=apply)


## Network layers: permutation-equivariance ##


def construct_symmetric_features(
    h_one: jnp.ndarray,
    h_two: jnp.ndarray,
    nspins: Tuple[int, int],
    h_aux: Optional[jnp.ndarray],
) -> jnp.ndarray:
  """Combines intermediate features from rank-one and -two streams.

  Args:
    h_one: set of one-electron features. Shape: (nelectrons, n1), where n1 is
      the output size of the previous layer.
    h_two: set of two-electron features. Shape: (nelectrons, nelectrons, n2),
      where n2 is the output size of the previous layer.
    nspins: Number of spin-up and spin-down electrons.
    h_aux: optional auxiliary features to include. Shape (nelectrons, naux).

  Returns:
    array containing the permutation-equivariant features: the input set of
    one-electron features, the mean of the one-electron features over each
    (occupied) spin channel, and the mean of the two-electron features over each
    (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2 + naux) if
    there are both spin-up and spin-down electrons and
    (nelectrons, 2*n1 + n2 + naux) otherwise.
  """
  # Split features into spin up and spin down electrons
  spin_partitions = network_blocks.array_partitions(nspins)
  h_ones = jnp.split(h_one, spin_partitions, axis=0)
  h_twos = jnp.split(h_two, spin_partitions, axis=0)

  # Construct inputs to next layer
  # h.size == 0 corresponds to unoccupied spin channels.
  g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
  g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]

  g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]

  features = [h_one] + g_one + g_two
  if h_aux is not None:
    features.append(h_aux)
  return jnp.concatenate(features, axis=1)


## Network layers: main layers ##


def make_schnet_convolution(
    nspins: Tuple[int, int], separate_spin_channels: bool
) -> ...:
  """Returns init/apply pair for SchNet-style convolutions.

  See Gerard et al, arXiv:2205.09438.

  Args:
    nspins: number of electrons in each spin channel.
    separate_spin_channels: If True, treat pairs of spin-parallel and
      spin-antiparallel electrons with separate  embeddings. If False, use the
      same embedding for all pairs.
  """

  def init(
      key: chex.PRNGKey, dims_one: int, dims_two: int, embedding_dim: int
  ) -> ParamTree:
    """Returns parameters for learned Schnet convolutions.

    Args:
      key: PRNG state.
      dims_one: number of hidden units of the one-electron layer.
      dims_two: number of hidden units of the two-electron layer.
      embedding_dim: embedding dimension to use for the convolution.
    """
    nchannels = 2 if separate_spin_channels else 1
    key_one, *key_two = jax.random.split(key, num=nchannels + 1)
    h_one_kernel = network_blocks.init_linear_layer(
        key_one, in_dim=dims_one, out_dim=embedding_dim, include_bias=False
    )
    h_two_kernels = []
    for i in range(nchannels):
      h_two_kernels.append(
          network_blocks.init_linear_layer(
              key_two[i],
              in_dim=dims_two,
              out_dim=embedding_dim,
              include_bias=False,
          )
      )
    return {
        'single': h_one_kernel['w'],
        'double': [kernel['w'] for kernel in h_two_kernels],
    }

  def apply(
      params: ParamTree, h_one: jnp.ndarray, h_two: Tuple[jnp.ndarray, ...]
  ) -> jnp.ndarray:
    """Applies the convolution B h_two . C h_one."""
    # Two distinctions from Gerard et al. They give the electron-electron
    # embedding in Eq 6 as
    # \sum_j B_{sigma_{ij}}(h_{ij} * C_{sigma_{ij}}(h_{j}
    # ie the C kernel is also dependent upon the spin pair. This does not match
    # the definition in the PauliNet paper. We keep the C kernel independent of
    # spin pair, and make B dependent upon spin-pair if separate_spin_channels
    # is True.
    # This (and Eq 5) gives that all j electrons are summed over, whereas
    # FermiNet concatenates the sum over spin up and spin-down electrons
    # separately. We follow the latter always.
    # These changes are in keeping with the spirit of FermiNet and SchNet
    # convolutions, if not the detail provided by Gerard et al.
    h_one_embedding = network_blocks.linear_layer(h_one, params['single'])
    h_two_embeddings = [
        network_blocks.linear_layer(h_two_channel, layer_param)
        for h_two_channel, layer_param in zip(h_two, params['double'])
    ]
    if separate_spin_channels:
      # h_two is a tuple of parallel spin pairs and anti-parallel spin pairs.
      h_two_embedding = _combine_spin_pairs(
          h_two_embeddings[0], h_two_embeddings[1], nspins
      )
    else:
      h_two_embedding = h_two_embeddings[0]
    return h_one_embedding * h_two_embedding

  return init, apply


def make_schnet_electron_nuclear_convolution() -> ...:
  """Returns init/apply pair for SchNet-style convolutions for electrons-ions.

  See Gerard et al, arXiv:2205.09438.
  """

  def init(
      key: chex.PRNGKey,
      electron_nuclear_dim: int,
      nuclear_dim: int,
      embedding_dim: int,
  ) -> Param:
    key1, key2 = jax.random.split(key)
    return {
        'electron_ion_embedding': network_blocks.init_linear_layer(
            key1,
            in_dim=electron_nuclear_dim,
            out_dim=embedding_dim,
            include_bias=False,
        )['w'],
        'ion_embedding': network_blocks.init_linear_layer(
            key2, in_dim=nuclear_dim, out_dim=embedding_dim, include_bias=False
        )['w'],
    }

  def apply(
      params: Param, h_ion_nuc: jnp.ndarray, nuc_embedding: jnp.ndarray
  ) -> jnp.ndarray:
    # h_ion_nuc is (nelec, natom, electron_nuclear_dim)
    # nuc_embedding is (natom, nuclear_dim)
    ion_nuc_conv = (h_ion_nuc @ params['electron_ion_embedding']) * (
        nuc_embedding[None] @ params['ion_embedding']
    )
    return jnp.sum(ion_nuc_conv, axis=1)

  return init, apply


def make_fermi_net_layers(
    nspins: Tuple[int, int], natoms: int, options: FermiNetOptions
) -> Tuple[InitLayersFn, ApplyLayersFn]:
  """Creates the permutation-equivariant and interaction layers for FermiNet.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    natoms: number of atoms.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """

  schnet_electron_init, schnet_electron_apply = make_schnet_convolution(
      nspins=nspins, separate_spin_channels=options.separate_spin_channels
  )
  schnet_electron_nuclear_init, schnet_electron_nuclear_apply = (
      make_schnet_electron_nuclear_convolution()
  )

  if all(
      len(hidden_dims) != len(options.hidden_dims[0])
      for hidden_dims in options.hidden_dims
  ):
    raise ValueError(
        'Each layer does not have the same number of streams: '
        f'{options.hidden_dims}'
    )

  if options.use_last_layer:
    num_convolutions = len(options.hidden_dims) + 1
  else:
    num_convolutions = len(options.hidden_dims)
  if (
      options.schnet_electron_electron_convolutions
      and len(options.schnet_electron_electron_convolutions) != num_convolutions
  ):
    raise ValueError(
        'Inconsistent number of layers for convolution and '
        'one- and two-electron streams. '
        f'{len(options.schnet_electron_electron_convolutions)=}, '
        f'expected {num_convolutions} layers.'
    )
  e_ion_options = (
      options.nuclear_embedding_dim,
      options.electron_nuclear_aux_dims,
      options.schnet_electron_nuclear_convolutions,
  )
  if any(e_ion_options) != all(e_ion_options):
    raise ValueError(
        'A subset of options set for electron-ion '
        'auxiliary stream: '
        f'{options.nuclear_embedding_dim=} '
        f'{options.electron_nuclear_aux_dims=} '
        f'{options.schnet_electron_nuclear_convolutions=}'
    )
  if (
      options.electron_nuclear_aux_dims
      and len(options.electron_nuclear_aux_dims) < num_convolutions - 1
  ):
    raise ValueError(
        'Too few layers in electron-nuclear auxiliary stream. '
        f'{options.electron_nuclear_aux_dims=}, '
        f'expected {num_convolutions-1} layers.'
    )
  if (
      options.schnet_electron_nuclear_convolutions
      and len(options.schnet_electron_nuclear_convolutions) != num_convolutions
  ):
    raise ValueError(
        'Inconsistent number of layers for convolution and '
        'one- and two-electron streams. '
        f'{len(options.schnet_electron_nuclear_convolutions)=}, '
        f'expected {num_convolutions} layers.'
    )

  def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""

    params = {}
    key, nuclear_key = jax.random.split(key, num=2)
    (num_one_features, num_two_features), params['input'] = (
        options.feature_layer.init()
    )
    if options.nuclear_embedding_dim:
      # Gerard et al project each nuclear charge to a separate vector.
      params['nuclear'] = network_blocks.init_linear_layer(
          nuclear_key,
          in_dim=1,
          out_dim=options.nuclear_embedding_dim,
          include_bias=True,
      )

    # The input to layer L of the one-electron stream is from
    # construct_symmetric_features and shape (nelectrons, nfeatures), where
    # nfeatures is
    # i) output from the previous one-electron layer (out1);
    # ii) the mean for each spin channel from each layer (out1 * # channels);
    # iii) the mean for each spin channel from each two-electron layer (out2 * #
    # channels)
    # iv) any additional features from auxiliary streams.
    # We don't create features for spin channels
    # which contain no electrons (i.e. spin-polarised systems).
    nchannels = len([nspin for nspin in nspins if nspin > 0])

    def nfeatures(out1, out2, aux):
      return (nchannels + 1) * out1 + nchannels * out2 + aux

    # one-electron stream, per electron:
    #  - one-electron features per atom (default: electron-atom vectors
    #    (ndim/atom) and distances (1/atom)),
    # two-electron stream, per pair of electrons:
    #  - two-electron features per electron pair (default: electron-electron
    #    vector (dim) and distance (1))
    dims_one_in = num_one_features
    dims_two_in = num_two_features
    # Note SchNet-style convolution with a electron-nuclear stream assumes
    # FermiNet features currently.
    dims_e_aux_in = num_one_features // natoms

    key, subkey = jax.random.split(key)
    layers = []
    for i in range(len(options.hidden_dims)):
      layer_params = {}
      key, single_key, *double_keys, aux_key = jax.random.split(key, num=5)

      # Learned convolution on each layer.
      if options.schnet_electron_electron_convolutions:
        key, subkey = jax.random.split(key)
        layer_params['schnet'] = schnet_electron_init(
            subkey,
            dims_one=dims_one_in,
            dims_two=dims_two_in,
            embedding_dim=options.schnet_electron_electron_convolutions[i],
        )
        dims_two_embedding = options.schnet_electron_electron_convolutions[i]
      else:
        dims_two_embedding = dims_two_in
      if options.schnet_electron_nuclear_convolutions:
        key, subkey = jax.random.split(key)
        layer_params['schnet_nuclear'] = schnet_electron_nuclear_init(
            subkey,
            electron_nuclear_dim=dims_e_aux_in,
            nuclear_dim=options.nuclear_embedding_dim,
            embedding_dim=options.schnet_electron_nuclear_convolutions[i],
        )
        dims_aux = options.schnet_electron_nuclear_convolutions[i]
      else:
        dims_aux = 0

      dims_one_in = nfeatures(dims_one_in, dims_two_embedding, dims_aux)

      # Layer initialisation
      dims_one_out, dims_two_out = options.hidden_dims[i]
      layer_params['single'] = network_blocks.init_linear_layer(
          single_key,
          in_dim=dims_one_in,
          out_dim=dims_one_out,
          include_bias=True,
      )

      if i < len(options.hidden_dims) - 1 or options.use_last_layer:
        ndouble_channels = 2 if options.separate_spin_channels else 1
        layer_params['double'] = []
        for ichannel in range(ndouble_channels):
          layer_params['double'].append(
              network_blocks.init_linear_layer(
                  double_keys[ichannel],
                  in_dim=dims_two_in,
                  out_dim=dims_two_out,
                  include_bias=True,
              )
          )
        if not options.separate_spin_channels:
          # Just have a single dict rather than a list of length 1 to match
          # older behaviour (when one stream was used for all electron pairs).
          layer_params['double'] = layer_params['double'][0]
        if options.electron_nuclear_aux_dims:
          layer_params['electron_ion'] = network_blocks.init_linear_layer(
              aux_key,
              in_dim=dims_e_aux_in,
              out_dim=options.electron_nuclear_aux_dims[i],
              include_bias=True,
          )
          dims_e_aux_in = options.electron_nuclear_aux_dims[i]

      layers.append(layer_params)
      dims_one_in = dims_one_out
      dims_two_in = dims_two_out

    if options.use_last_layer:
      layers.append({})
      # Pass symmetric features to the orbital shaping layer.
      if options.schnet_electron_electron_convolutions:
        key, subkey = jax.random.split(key)
        layers[-1]['schnet'] = schnet_electron_init(
            subkey,
            dims_one=dims_one_in,
            dims_two=dims_two_in,
            embedding_dim=options.schnet_electron_electron_convolutions[-1],
        )
        dims_two_in = options.schnet_electron_electron_convolutions[-1]
      if options.schnet_electron_nuclear_convolutions:
        key, subkey = jax.random.split(key)
        layers[-1]['schnet_nuclear'] = schnet_electron_nuclear_init(
            subkey,
            electron_nuclear_dim=dims_e_aux_in,
            nuclear_dim=options.nuclear_embedding_dim,
            embedding_dim=options.schnet_electron_nuclear_convolutions[-1],
        )
        dims_aux = options.schnet_electron_nuclear_convolutions[-1]
      else:
        dims_aux = 0
      output_dims = nfeatures(dims_one_in, dims_two_in, dims_aux)
    else:
      # Pass output of the one-electron stream straight to orbital shaping.
      output_dims = dims_one_in

    params['streams'] = layers

    return output_dims, params

  def electron_electron_convolution(
      params: ParamTree,
      h_one: jnp.ndarray,
      h_two: Tuple[jnp.ndarray, ...],
  ) -> jnp.ndarray:
    if options.schnet_electron_electron_convolutions:
      # SchNet-style embedding: convolve embeddings of one- and two-electron
      # streams.
      h_two_embedding = schnet_electron_apply(params['schnet'], h_one, h_two)
    elif options.separate_spin_channels:
      # FermiNet embedding from separate spin channels for parallel and
      # anti-parallel pairs of spins. Need to reshape and combine spin channels.
      h_two_embedding = _combine_spin_pairs(h_two[0], h_two[1], nspins)
    else:
      # Original FermiNet embedding.
      h_two_embedding = h_two[0]
    return h_two_embedding

  def apply_layer(
      params: Mapping[str, ParamTree],
      h_one: jnp.ndarray,
      h_two: Tuple[jnp.ndarray, ...],
      h_elec_ion: Optional[jnp.ndarray],
      nuclear_embedding: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...], Optional[jnp.ndarray]]:
    if options.separate_spin_channels:
      assert len(h_two) == 2
    else:
      assert len(h_two) == 1

    residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y

    # Permutation-equivariant block.
    h_two_embedding = electron_electron_convolution(params, h_one, h_two)
    if options.schnet_electron_nuclear_convolutions:
      h_aux = schnet_electron_nuclear_apply(
          params['schnet_nuclear'], h_elec_ion, nuclear_embedding
      )
    else:
      h_aux = None
    h_one_in = construct_symmetric_features(
        h_one, h_two_embedding, nspins, h_aux=h_aux
    )

    # Execute next layer.
    h_one_next = jnp.tanh(
        network_blocks.linear_layer(h_one_in, **params['single'])
    )
    h_one = residual(h_one, h_one_next)
    # Only perform the auxiliary streams if parameters are present (ie not the
    # final layer of the network if use_last_layer is False).
    if 'double' in params:
      if options.separate_spin_channels:
        params_double = params['double']
      else:
        # Using one stream for pairs of electrons. Make a sequence of params of
        # same length as h_two.
        params_double = [params['double']]
      h_two_next = [
          jnp.tanh(network_blocks.linear_layer(prev, **param))
          for prev, param in zip(h_two, params_double)
      ]
      h_two = tuple(residual(prev, new) for prev, new in zip(h_two, h_two_next))
    if h_elec_ion is not None and 'electron_ion' in params:
      h_elec_ion = network_blocks.linear_layer(
          h_elec_ion, **params['electron_ion']
      )

    return h_one, h_two, h_elec_ion

  def apply(
      params,
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the FermiNet interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permutation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """
    del spins  # Unused.

    ae_features, ee_features = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )
    #print("ae_features", ae_features.shape)
    #print("ee_features", ee_features.shape)

    if options.electron_nuclear_aux_dims:
      # Electron-ion auxiliary stream just takes electron-ion vectors and
      # distances.
      h_elec_ion = jnp.reshape(ae_features, (ae_features.shape[0], natoms, -1))
    else:
      h_elec_ion = None

    h_one = ae_features  # single-electron features

    if options.separate_spin_channels:
      # Use the same stream for spin-parallel and spin-antiparallel electrons.
      # In order to handle different numbers of spin-up and spin-down electrons,
      # flatten the i,j indices.
      # Shapes: (nup*nup + ndown*ndown, nfeatures), (nup*down*2, nfeatures).
      h_two = _split_spin_pairs(ee_features, nspins)
    else:
      # Use the same stream for spin-parallel and spin-antiparallel electrons.
      # Keep as 3D array to make splitting over spin channels in
      # construct_symmetric_features simple.
      # Shape: (nelectron, nelectron, nfeatures)
      h_two = [ee_features]

    if options.nuclear_embedding_dim:
      nuclear_embedding = network_blocks.linear_layer(
          charges[:, None], **params['nuclear']
      )
    else:
      nuclear_embedding = None

    for i in range(len(options.hidden_dims)):
      h_one, h_two, h_elec_ion = apply_layer(
          params['streams'][i],
          h_one,
          h_two,
          h_elec_ion,
          nuclear_embedding,
      )

    if options.use_last_layer:
      last_layer = params['streams'][-1]
      h_two_embedding = electron_electron_convolution(last_layer, h_one, h_two)
      if options.schnet_electron_nuclear_convolutions:
        h_aux = schnet_electron_nuclear_apply(
            last_layer['schnet_nuclear'], h_elec_ion, nuclear_embedding
        )
      else:
        h_aux = None
      h_to_orbitals = construct_symmetric_features(
          h_one, h_two_embedding, nspins, h_aux=h_aux
      )
    else:
      # Didn't apply the final two-electron and auxiliary layers. Just forward
      # the output of the one-electron stream to the orbital projection layer.
      h_to_orbitals = h_one

    return h_to_orbitals

  return init, apply


## Network layers: orbitals ##


def make_mlp() ->...:
  """Construct MLP, with final linear projection to embedding size."""

  def init(key: chex.PRNGKey, mlp_hidden_dims: Tuple[int, ...],
      embed_dim: int, out_dim: int) -> Sequence[Param]:
    params = []
    dims_one_in = [embed_dim, *mlp_hidden_dims]
    #dims_one_out = [*mlp_hidden_dims, embed_dim]
    dims_one_out = [*mlp_hidden_dims, out_dim]
    n_layer = len(dims_one_in)
    for i in range(n_layer-1):
      key, subkey = jax.random.split(key)
      params.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_one_in[i],
              out_dim=dims_one_out[i],
              include_bias=True))
    for i in range(n_layer-1, n_layer):
      key, subkey = jax.random.split(key)
      params.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_one_in[i],
              out_dim=dims_one_out[i],
              include_bias=False))
    return params

  def apply(params: Sequence[Param],
            inputs: jnp.ndarray) -> jnp.ndarray:
    x = inputs
    for i in range(len(params)-1):
      #x = jnp.tanh(network_blocks.linear_layer(x, **params[i]))
      x = jax.nn.gelu(network_blocks.linear_layer(x, **params[i]))
    for i in range(len(params)-1,len(params)):
      x = network_blocks.linear_layer(x, **params[i])
    return x

  return init, apply


def make_sobf_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for spin orbitals with layer spin selection, using mlp as oribtials, without jastrow and pre-orbitals.
  This only apply to Psiformer!!

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  mlp_up_init, mlp_up_apply = make_mlp()
  mlp_down_init, mlp_down_apply = make_mlp()

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    #nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det: # this is true for Psiformer
        # Dense determinant. Need N orbitals per electron per determinant.
        #norbitals = sum(nspins) * options.determinants * num_states
        #norbitals = nkpt * options.determinants * num_states
        norbitals = sum(nspins) * options.determinants * num_states 
      #else:
      #  # Spin-factored block-diagonal determinant. Need nspin orbitals per
      #  # electron per determinant.
      #  orbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)
    #print("nspin_orbitals", nspin_orbitals)

    ## create envelope params
    #natom = charges.shape[0]
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  # Applied to output from final layer of 1e stream.
    #  output_dims = dims_orbital_in
    #elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  # Applied to orbitals.
    #  if options.complex_output:
    #    output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
    #  else:
    #    output_dims = nspin_orbitals
    #else:
    #  raise ValueError('Unknown envelope type')
    #params['envelope'] = options.envelope.init(
    #    natom=natom, output_dims=output_dims, ndim=options.ndim
    #)

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    ## orbital shaping
    #orbitals_up = []
    #orbitals_down = []
    #for nspin_orbital in nspin_orbitals:
    #  key, subkey = jax.random.split(key)
    #  orbitals_up.append(
    #      network_blocks.init_linear_layer(
    #          subkey,
    #          in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
    #          out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
    #          include_bias=options.bias_orbitals,
    #      )
    #  )
    #for nspin_orbital in nspin_orbitals:
    #  key, subkey = jax.random.split(key)
    #  orbitals_down.append(
    #      network_blocks.init_linear_layer(
    #          subkey,
    #          in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
    #          out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
    #          include_bias=options.bias_orbitals,
    #      )
    #  )
    #params['orbital_up'] = orbitals_up
    #params['orbital_down'] = orbitals_down

    key, mlp_key1 = jax.random.split(key)
    key, mlp_key2 = jax.random.split(key)
    mlp_hidden_dims = [options.system.hdim, options.system.hdim]
    attn_dim = options.system.attn_dim
    params['mlp_up'] = mlp_up_init(mlp_key1, mlp_hidden_dims, attn_dim, options.system.ndet*2*n_par)
    params['mlp_down'] = mlp_down_init(mlp_key2, mlp_hidden_dims, attn_dim, options.system.ndet*2*n_par)

    #key, key1 = jax.random.split(key)
    #key, key2 = jax.random.split(key)
    #initializer = jax.nn.initializers.lecun_normal()
    #params['cmat_real'] = initializer(key1, shape=(nkpt, n_par, ndet)) + jnp.eye(nkpt, n_par)[...,None]  
    #params['cmat_imag'] = initializer(key2, shape=(nkpt, n_par, ndet)) 

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  envelope_factor = options.envelope.apply(
    #      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    #  )
    #  h_to_orbitals = envelope_factor * h_to_orbitals

    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    #for i in range(len(h_to_orbitals)): print("h_to_orbitals i", i, h_to_orbitals[i].shape) # (n_par, n_hdim=num_heads*heads_dim)

    ##active_spin_channels = [spin for spin in nspins if spin > 0]
    ##active_spin_partitions = network_blocks.array_partitions(
    ##    active_spin_channels
    ##)

    # Create orbitals.
    orbitals_up = [
        #network_blocks.linear_layer(h, **p)
        #mlp_up_apply(h, **p)
        #for h, p in zip(h_to_orbitals, params['mlp_up'])
        mlp_up_apply(params['mlp_up'], h_to_orbitals[0])
    ]
    orbitals_down = [
        #network_blocks.linear_layer(h, **p)
        #mlp_down_apply(h, **p)
        #for h, p in zip(h_to_orbitals, params['mlp_down'])
        mlp_down_apply(params['mlp_down'], h_to_orbitals[0])
    ]
    #for i in range(len(orbitals_up)): print("orbitals_up i", i, orbitals_up[i].shape) # (n_par, n_par*ndet*2)
    #for i in range(len(orbitals_down)): print("orbitals_down i", i, orbitals_down[i].shape) # (n_par, n_par*ndet*2)
    if options.complex_output:
      # create imaginary orbitals
      orbitals_up = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals_up
      ]
      orbitals_down = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals_down
      ]
    #for i in range(len(orbitals_up)): print("orbitals_up i", i, orbitals_up[i].shape) # (n_par, n_par*ndet)
    #for i in range(len(orbitals_down)): print("orbitals_down i", i, orbitals_down[i].shape) # (n_par, n_par*ndet)

    # Reshape into matrices.
    shapes = [(n_par, -1, n_par)]
    #print("shapes",shapes)
    orbitals_up = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals_up, shapes)
    ]
    orbitals_down = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals_down, shapes)
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, ndet, n_par)
    orbitals_up = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals_up]
    orbitals_down = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals_down]
    if options.full_det: # true for Psiformer
      orbitals_up = [jnp.concatenate(orbitals_up, axis=1)]
      orbitals_down = [jnp.concatenate(orbitals_down, axis=1)]
    #for i in range(len(orbitals_up)): print("orbitals_up i", i, orbitals_up[i].shape) # (n_par, nkpt*ndet*2)
    #for i in range(len(orbitals_down)): print("orbitals_down i", i, orbitals_down[i].shape) # (n_par, nkpt*ndet*2)

    orbital_up = orbitals_up[0]
    orbital_down = orbitals_down[0]
    spin_mask = jnp.array((spins+1)/2, dtype=int)
    #print("spin_mask", spin_mask.shape)
    #print("orbital_up", orbital_up.shape)
    #print("orbital_down", orbital_down.shape)
    orb = jnp.where(spin_mask[None,:, None] == 1, orbital_up, orbital_down)
    orbitals = [orb]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, n_par)

    # Optionally apply Jastrow factor for electron cusp conditions.
    # Added pre-determinant for compatibility with pretraining.
    # Warning: this jastrow takes r_ee directly, which could violate PBC!!!
    #if jastrow_apply is not None:
    #  jastrow = jnp.exp(
    #      jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
    #  )
    #  orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply


def make_hfbfad_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals with hartree fock backflow, without jastrow and pre-orbitals.
  sum_w w det(v^w + uk^w(ri;r)) where v^w and w(ri;r) are from Psiformer

  This only apply to Psiformer!!

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  mlp_real_init, mlp_real_apply = make_mlp()
  mlp_imag_init, mlp_imag_apply = make_mlp()

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det: # this is true for Psiformer
        # Dense determinant. Need N orbitals per electron per determinant.
        #norbitals = sum(nspins) * options.determinants * num_states
        norbitals = nkpt * options.determinants * num_states
      #else:
      #  # Spin-factored block-diagonal determinant. Need nspin orbitals per
      #  # electron per determinant.
      #  orbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)
    #print("nspin_orbitals", nspin_orbitals)

    # create envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # orbital shaping
    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
              #out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
              out_dim=sdim*2*ndet, # n_par*n_det*2 for complex orbital
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    key, key1 = jax.random.split(key)
    key, key2 = jax.random.split(key)
    initializer = jax.nn.initializers.lecun_normal()
    params['cmat_real'] = initializer(key1, shape=(ndet, nkpt, n_par))  
    params['cmat_imag'] = initializer(key2, shape=(ndet, nkpt, n_par)) 

    key, mlp_key1 = jax.random.split(key)
    key, mlp_key2 = jax.random.split(key)
    mlp_hidden_dims = [options.system.hdim, options.system.hdim]
    attn_dim = options.system.attn_dim
    params['mlp_real'] = mlp_real_init(mlp_key1, mlp_hidden_dims, attn_dim+ngpt*2, options.system.ndet)
    params['mlp_imag'] = mlp_imag_init(mlp_key2, mlp_hidden_dims, attn_dim+ngpt*2, options.system.ndet)

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """


    ############################## basic features ##########################################
    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet
    vu_arr = options.system.vs_arr
    kp = options.system.kp
    km = options.system.km
    kp_vec = options.system.kp_vec # (nkpt, ngpt, sdim)
    km_vec = options.system.km_vec
    #print("vu_arr", vu_arr.shape) # (nkpt, ngpt*2)
    spin_mask = jnp.array((spins+1)/2, dtype=int)

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    ############################### create backflow and orbs ##########################################
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  envelope_factor = options.envelope.apply(
    #      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    #  )
    #  h_to_orbitals = envelope_factor * h_to_orbitals

    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    #for i in range(len(h_to_orbitals)): print("h_to_orbitals i", i, h_to_orbitals[i].shape) # (n_par, n_hdim=num_heads*heads_dim)

    #active_spin_channels = [spin for spin in nspins if spin > 0]
    #active_spin_partitions = network_blocks.array_partitions(
    #    active_spin_channels
    #)

    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, sdim*ndet*2)
    if options.complex_output:
      # create imaginary orbitals
      orbitals = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
      ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, sdim*ndet)

    # Apply envelopes if required.
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    #  r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    #  r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    #  for i in range(len(active_spin_channels)):
    #    orbitals[i] = orbitals[i] * options.envelope.apply(
    #        ae=ae_channels[i],
    #        r_ae=r_ae_channels[i],
    #        r_ee=r_ee_channels[i],
    #        **params['envelope'][i],
    #    )

    # Reshape into matrices.
    shapes = [(n_par, -1, sdim)]
    #print("shapes",shapes)
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, ndet, sdim)
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det: # true for Psiformer
      orbitals = [jnp.concatenate(orbitals, axis=1)]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, sdim)

    bf = orbitals[0] # backflow transformation, (ndet, n_par, sdim)
    #print("bf", bf.shape)
    cmat = params["cmat_real"] + 1j*params["cmat_imag"] + jnp.eye(nkpt, n_par)[None,...] # (ndet, nkpt, n_par)
    #orbitals = [bf]

    ### select k_vec
    kp_vec2 = kp_vec.reshape(nkpt,-1) # (nkpt, ngpt*sdim)
    #print("kp_vec2", kp_vec2.shape)
    km_vec2 = km_vec.reshape(nkpt,-1) # (nkpt, ngpt*sdim)
    kp_vec2 = jnp.tile(kp_vec2[None,...,:], (n_par,1,1))
    #print("kp_vec2", kp_vec2.shape)
    km_vec2 = jnp.tile(km_vec2[None,...,:], (n_par,1,1))
    #print('kp_vec',kp_vec.shape) # (n_par,nkpt,ng_f*sdim)
    condition = jnp.tile(spin_mask[:, None, None], (1,nkpt,ngpt*sdim)) # 1 is spin up, 0(-1) is spin down
    #print('condition',condition.shape)
    k_vec = jnp.where(condition, kp_vec2, km_vec2) # (n_par,nkpt,ngpt*sdim)
    #print("k_vec", k_vec.shape)

    g_orbs = jnp.tile(h_to_orbitals[0][...,None,:], (1,nkpt,1))
    #print('g_orbs',g_orbs.shape) # (n_par,nkpt,attn)
    temp = jnp.concatenate((g_orbs, k_vec), axis=-1)
    #print('temp',temp.shape) # (n_par,nkpt,attn+ngpt*sdim=attn+49*2=130)
    orbs_r = mlp_real_apply(params['mlp_real'], temp) # jastrow, (n_par,n_par,ndet) before squeeze
    orbs_i = mlp_imag_apply(params['mlp_imag'], temp) # jastrow, (n_par,n_par,ndet) before squeeze
    j_orbs = orbs_r + 1j*orbs_i # (n_par,nkpt, ndet)
    #print("j_orbs", j_orbs.shape)


    ############################# parallel version ########################################

    #print("pos", pos.shape) # (n_par, sdim)
    x_bf = pos.reshape(n_par,sdim)[None,...] + 0.5*bf # (ndet, n_par, sdim)
    #print("x_bf", x_bf.shape)
    x_bf = x_bf.reshape(ndet*n_par,sdim) # (ndet*n_par, sdim)
    #print("x_bf", x_bf.shape)

    # just need a function that takes ae as input and output orbitals of shape [1,n_par,n_par]
    # kp_vec is of shape (nkpt, ngpt, sdim)
    #eikpr = jnp.exp(1j*jnp.dot(kp_vec+kp,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par) # exp(i(kp+k+g)r)
    #eikmr = jnp.exp(1j*jnp.dot(km_vec+km,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par) # exp(i(km+k+g)r)
    eikpr = jnp.exp(1j*jnp.dot(kp_vec+kp,x_bf.T)) # (nkpt,ngpt,ndet*n_par) # exp(i(kp+k+g)r)
    eikmr = jnp.exp(1j*jnp.dot(km_vec+km,x_bf.T)) # (nkpt,ngpt,ndet*n_par) # exp(i(km+k+g)r)
    #print("eikpr",eikpr.shape)
    #print("eikmr",eikmr.shape)

    #ueikpr = jnp.sum(jnp.multiply(vu_arr[:,:ngpt][:, :, None], eikpr), axis=1)  # (nkpt,ndet*n_par)
    #ueikmr = jnp.sum(jnp.multiply(vu_arr[:,ngpt:][:, :, None], eikmr), axis=1)  # (nkpt,ndet*n_par)
    ueikpr = jnp.matmul(vu_arr[:,:ngpt][:, None, :], eikpr).squeeze(1)  # (nkpt,ndet*n_par)
    ueikmr = jnp.matmul(vu_arr[:,ngpt:][:, None, :], eikmr).squeeze(1)  # (nkpt,ndet*n_par)
    ueikpr = ueikpr.reshape(nkpt,ndet,n_par).transpose(1,2,0) # (ndet, n_par, nkpt)
    ueikmr = ueikmr.reshape(nkpt,ndet,n_par).transpose(1,2,0) # (ndet, n_par, nkpt)
    #tmp2 = jnp.sum(jnp.multiply(vu_arr[:nk_par,ngpt:][:, :, None], tmp), axis=1)  # (nkpt,n_par)
    #print("ueikmr",ueikmr.shape)
    #print("ueikpr",ueikpr.shape)
    #jax.debug.print('{}', jnp.linalg.norm(jnp.abs(ueikmr)-jnp.abs(tmp2)))

    orb = jnp.where(spin_mask[None,:,None] == 1, ueikpr, ueikmr) # (ndet, n_par, nkpt)
    #tmp3 = jnp.where(spin_mask[None,:] == 1, ueikpr, tmp2)
    #jax.debug.print('{}', jnp.linalg.norm(jnp.abs(orb)-jnp.abs(tmp3)))
    #print("spin_mask",spin_mask.shape)
    #print("orbitals",orb.shape)
    orb = orb + 1.0*j_orbs.transpose(2,0,1) # (ndet, n_par, nkpt)
    orb = jnp.matmul(orb, cmat) # (ndet, n_par, n_par)
    #orb = jnp.expand_dims(orb,axis=0)
    #print("orb",orb.shape)
    orbitals = [orb]
    #for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (ndet,nup,nup)


    ############################### for loop version ########################################
    #x_bf = pos.reshape(-1,sdim)[None,...] + 0.5*bf # (ndet, n_par, sdim)
    #orbitals_list = []
    #for i in range(ndet):
    #  mat = []
    #  for j in range(nkpt):
    #    xp = jnp.exp(1j*jnp.dot((kp_vec+kp)[j,:,:],x_bf[i].T)) # (ngpt,n_par)
    #    uxp = vu_arr[j,:ngpt] @ xp # (n_par,)
    #    xm = jnp.exp(1j*jnp.dot((km_vec+km)[j,:,:],x_bf[i].T)) # (ngpt,n_par)
    #    uxm = vu_arr[j,ngpt:] @ xm # (n_par,)
    #    ux = jnp.where(spin_mask,uxp,uxm) # (n_par)
    #    #print("ux", ux.shape)
    #    mat.append(ux)
    #  mat = jnp.stack(mat, axis=0) # (n_par,nkpt) 
    #  #print("mat", mat.shape)
    #  mat = cmat[i].T @ mat # (n_par, n_par)
    #  orbitals_list.append(mat)

    #orbitals_list = jnp.stack(orbitals_list,axis=0)
    ##print("orbitals_list", orbitals_list.shape)
    #orbitals = [orbitals_list]


    return orbitals

  return init, apply


def make_hfbf_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals with hartree fock backflow, without jastrow and pre-orbitals.
  sum_w w det(v^w uk^w(ri;r)) where v^w and w(ri;r) are from Psiformer

  This only apply to Psiformer!!

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  mlp_real_init, mlp_real_apply = make_mlp()
  mlp_imag_init, mlp_imag_apply = make_mlp()

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det: # this is true for Psiformer
        # Dense determinant. Need N orbitals per electron per determinant.
        #norbitals = sum(nspins) * options.determinants * num_states
        norbitals = nkpt * options.determinants * num_states
      #else:
      #  # Spin-factored block-diagonal determinant. Need nspin orbitals per
      #  # electron per determinant.
      #  orbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)
    #print("nspin_orbitals", nspin_orbitals)

    # create envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # orbital shaping
    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
              #out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
              out_dim=sdim*2*ndet, # n_par*n_det*2 for complex orbital
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    key, key1 = jax.random.split(key)
    key, key2 = jax.random.split(key)
    initializer = jax.nn.initializers.lecun_normal()
    params['cmat_real'] = initializer(key1, shape=(ndet, nkpt, n_par))  
    params['cmat_imag'] = initializer(key2, shape=(ndet, nkpt, n_par)) 

    key, mlp_key1 = jax.random.split(key)
    key, mlp_key2 = jax.random.split(key)
    mlp_hidden_dims = [options.system.hdim, options.system.hdim]
    attn_dim = options.system.attn_dim
    params['mlp_real'] = mlp_real_init(mlp_key1, mlp_hidden_dims, attn_dim+ngpt*2, options.system.ndet)
    params['mlp_imag'] = mlp_imag_init(mlp_key2, mlp_hidden_dims, attn_dim+ngpt*2, options.system.ndet)

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """


    ############################## basic features ##########################################
    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet
    vu_arr = options.system.vs_arr
    kp = options.system.kp
    km = options.system.km
    kp_vec = options.system.kp_vec # (nkpt, ngpt, sdim)
    km_vec = options.system.km_vec
    #print("vu_arr", vu_arr.shape) # (nkpt, ngpt*2)
    spin_mask = jnp.array((spins+1)/2, dtype=int)

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    ############################### create backflow and orbs ##########################################
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  envelope_factor = options.envelope.apply(
    #      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    #  )
    #  h_to_orbitals = envelope_factor * h_to_orbitals

    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    #for i in range(len(h_to_orbitals)): print("h_to_orbitals i", i, h_to_orbitals[i].shape) # (n_par, n_hdim=num_heads*heads_dim)

    #active_spin_channels = [spin for spin in nspins if spin > 0]
    #active_spin_partitions = network_blocks.array_partitions(
    #    active_spin_channels
    #)

    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, sdim*ndet*2)
    if options.complex_output:
      # create imaginary orbitals
      orbitals = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
      ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, sdim*ndet)

    # Apply envelopes if required.
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    #  r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    #  r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    #  for i in range(len(active_spin_channels)):
    #    orbitals[i] = orbitals[i] * options.envelope.apply(
    #        ae=ae_channels[i],
    #        r_ae=r_ae_channels[i],
    #        r_ee=r_ee_channels[i],
    #        **params['envelope'][i],
    #    )

    # Reshape into matrices.
    shapes = [(n_par, -1, sdim)]
    #print("shapes",shapes)
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, ndet, sdim)
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det: # true for Psiformer
      orbitals = [jnp.concatenate(orbitals, axis=1)]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, sdim)

    bf = orbitals[0] # backflow transformation, (ndet, n_par, sdim)
    #print("bf", bf.shape)
    cmat = params["cmat_real"] + 1j*params["cmat_imag"] + jnp.eye(nkpt, n_par)[None,...] # (ndet, nkpt, n_par)
    #orbitals = [bf]

    ### select k_vec
    kp_vec2 = kp_vec.reshape(nkpt,-1) # (nkpt, ngpt*sdim)
    #print("kp_vec2", kp_vec2.shape)
    km_vec2 = km_vec.reshape(nkpt,-1) # (nkpt, ngpt*sdim)
    kp_vec2 = jnp.tile(kp_vec2[None,...,:], (n_par,1,1))
    #print("kp_vec2", kp_vec2.shape)
    km_vec2 = jnp.tile(km_vec2[None,...,:], (n_par,1,1))
    #print('kp_vec',kp_vec.shape) # (n_par,nkpt,ng_f*sdim)
    condition = jnp.tile(spin_mask[:, None, None], (1,nkpt,ngpt*sdim)) # 1 is spin up, 0(-1) is spin down
    #print('condition',condition.shape)
    k_vec = jnp.where(condition, kp_vec2, km_vec2) # (n_par,nkpt,ngpt*sdim)
    #print("k_vec", k_vec.shape)

    g_orbs = jnp.tile(h_to_orbitals[0][...,None,:], (1,nkpt,1))
    #print('g_orbs',g_orbs.shape) # (n_par,nkpt,attn)
    temp = jnp.concatenate((g_orbs, k_vec), axis=-1)
    #print('temp',temp.shape) # (n_par,nkpt,attn+ngpt*sdim=attn+49*2=130)
    orbs_r = mlp_real_apply(params['mlp_real'], temp) # jastrow, (n_par,n_par,ndet) before squeeze
    orbs_i = mlp_imag_apply(params['mlp_imag'], temp) # jastrow, (n_par,n_par,ndet) before squeeze
    j_orbs = orbs_r + 1j*orbs_i # (n_par,nkpt, ndet)
    #print("j_orbs", j_orbs.shape)


    ############################ parallel version ########################################
    #print("pos", pos.shape) # (n_par, sdim)
    x_bf = pos.reshape(n_par,sdim)[None,...] + 0.5*bf # (ndet, n_par, sdim)
    #print("x_bf", x_bf.shape)
    x_bf = x_bf.reshape(ndet*n_par,sdim) # (ndet*n_par, sdim)
    #print("x_bf", x_bf.shape)

    # just need a function that takes ae as input and output orbitals of shape [1,n_par,n_par]
    # kp_vec is of shape (nkpt, ngpt, sdim)
    #eikpr = jnp.exp(1j*jnp.dot(kp_vec+kp,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par) # exp(i(kp+k+g)r)
    #eikmr = jnp.exp(1j*jnp.dot(km_vec+km,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par) # exp(i(km+k+g)r)
    eikpr = jnp.exp(1j*jnp.dot(kp_vec+kp,x_bf.T)) # (nkpt,ngpt,ndet*n_par) # exp(i(kp+k+g)r)
    eikmr = jnp.exp(1j*jnp.dot(km_vec+km,x_bf.T)) # (nkpt,ngpt,ndet*n_par) # exp(i(km+k+g)r)
    #print("eikpr",eikpr.shape)
    #print("eikmr",eikmr.shape)

    #ueikpr = jnp.sum(jnp.multiply(vu_arr[:,:ngpt][:, :, None], eikpr), axis=1)  # (nkpt,ndet*n_par)
    #ueikmr = jnp.sum(jnp.multiply(vu_arr[:,ngpt:][:, :, None], eikmr), axis=1)  # (nkpt,ndet*n_par)
    ueikpr = jnp.matmul(vu_arr[:,:ngpt][:, None, :], eikpr).squeeze(1)  # (nkpt,ndet*n_par)
    ueikmr = jnp.matmul(vu_arr[:,ngpt:][:, None, :], eikmr).squeeze(1)  # (nkpt,ndet*n_par)
    ueikpr = ueikpr.reshape(nkpt,ndet,n_par).transpose(1,2,0) # (ndet, n_par, nkpt)
    ueikmr = ueikmr.reshape(nkpt,ndet,n_par).transpose(1,2,0) # (ndet, n_par, nkpt)
    #tmp2 = jnp.sum(jnp.multiply(vu_arr[:nk_par,ngpt:][:, :, None], tmp), axis=1)  # (nkpt,n_par)
    #print("ueikmr",ueikmr.shape)
    #print("ueikpr",ueikpr.shape)
    #jax.debug.print('{}', jnp.linalg.norm(jnp.abs(ueikmr)-jnp.abs(tmp2)))

    orb = jnp.where(spin_mask[None,:,None] == 1, ueikpr, ueikmr) # (ndet, n_par, nkpt)
    #tmp3 = jnp.where(spin_mask[None,:] == 1, ueikpr, tmp2)
    #jax.debug.print('{}', jnp.linalg.norm(jnp.abs(orb)-jnp.abs(tmp3)))
    #print("spin_mask",spin_mask.shape)
    #print("orbitals",orb.shape)
    orb = orb * j_orbs.transpose(2,0,1) # (ndet, n_par, nkpt)
    orb = jnp.matmul(orb, cmat) # (ndet, n_par, n_par)
    #orb = jnp.expand_dims(orb,axis=0)
    #print("orb",orb.shape)
    orbitals = [orb]
    #for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (ndet,nup,nup)


    ################################ for loop version ########################################
    #x_bf = pos.reshape(-1,sdim)[None,...] + 0.5*bf # (ndet, n_par, sdim)
    #orbitals_list = []
    #for i in range(ndet):
    #  mat = []
    #  for j in range(nkpt):
    #    xp = jnp.exp(1j*jnp.dot((kp_vec+kp)[j,:,:],x_bf[i].T)) # (ngpt,n_par)
    #    uxp = vu_arr[j,:ngpt] @ xp # (n_par,)
    #    xm = jnp.exp(1j*jnp.dot((km_vec+km)[j,:,:],x_bf[i].T)) # (ngpt,n_par)
    #    uxm = vu_arr[j,ngpt:] @ xm # (n_par,)
    #    ux = jnp.where(spin_mask,uxp,uxm) # (n_par)
    #    #print("ux", ux.shape)
    #    mat.append(ux)
    #  mat = jnp.stack(mat, axis=0) # (nkpt, n_par) 
    #  #print("mat", mat.shape)
    #  mat = j_orbs[...,i].T * mat
    #  mat = cmat[i].T @ mat # (n_par, n_par)
    #  orbitals_list.append(mat)

    #orbitals_list = jnp.stack(orbitals_list,axis=0)
    ###print("orbitals_list", orbitals_list.shape)
    #orbitals = [orbitals_list]

    return orbitals

  return init, apply


def make_so_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for spin orbitals with layer spin selection, without jastrow and pre-orbitals.
  This only apply to Psiformer!!

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    #nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det: # this is true for Psiformer
        # Dense determinant. Need N orbitals per electron per determinant.
        #norbitals = sum(nspins) * options.determinants * num_states
        #norbitals = nkpt * options.determinants * num_states
        norbitals = sum(nspins) * options.determinants * num_states 
      #else:
      #  # Spin-factored block-diagonal determinant. Need nspin orbitals per
      #  # electron per determinant.
      #  orbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)
    #print("nspin_orbitals", nspin_orbitals)

    # create envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # orbital shaping
    orbitals_up = []
    orbitals_down = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals_up.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
              out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
              include_bias=options.bias_orbitals,
          )
      )
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals_down.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
              out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital_up'] = orbitals_up
    params['orbital_down'] = orbitals_down

    #key, key1 = jax.random.split(key)
    #key, key2 = jax.random.split(key)
    #initializer = jax.nn.initializers.lecun_normal()
    #params['cmat_real'] = initializer(key1, shape=(nkpt, n_par, ndet)) + jnp.eye(nkpt, n_par)[...,None]  
    #params['cmat_imag'] = initializer(key2, shape=(nkpt, n_par, ndet)) 

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  envelope_factor = options.envelope.apply(
    #      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    #  )
    #  h_to_orbitals = envelope_factor * h_to_orbitals

    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    #for i in range(len(h_to_orbitals)): print("h_to_orbitals i", i, h_to_orbitals[i].shape) # (n_par, n_hdim=num_heads*heads_dim)

    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(
        active_spin_channels
    )

    # Create orbitals.
    orbitals_up = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital_up'])
    ]
    orbitals_down = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital_down'])
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, nkpt*ndet*2)
    if options.complex_output:
      # create imaginary orbitals
      orbitals_up = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals_up
      ]
      orbitals_down = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals_down
      ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, nkpt*ndet)

    # Apply envelopes if required.
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    #  r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    #  r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    #  for i in range(len(active_spin_channels)):
    #    orbitals[i] = orbitals[i] * options.envelope.apply(
    #        ae=ae_channels[i],
    #        r_ae=r_ae_channels[i],
    #        r_ee=r_ee_channels[i],
    #        **params['envelope'][i],
    #    )

    # Reshape into matrices.
    shapes = [
        (spin, -1, sum(nspins) if options.full_det else spin)
        for spin in active_spin_channels
    ]
    #print("shapes",shapes)
    orbitals_up = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals_up, shapes)
    ]
    orbitals_down = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals_down, shapes)
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, ndet, n_par)
    orbitals_up = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals_up]
    orbitals_down = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals_down]
    if options.full_det: # true for Psiformer
      orbitals_up = [jnp.concatenate(orbitals_up, axis=1)]
      orbitals_down = [jnp.concatenate(orbitals_down, axis=1)]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, n_par)

    orbital_up = orbitals_up[0]
    orbital_down = orbitals_down[0]
    spin_mask = jnp.array((spins+1)/2, dtype=int)
    #print("spin_mask", spin_mask.shape)
    #print("orbital_up", orbital_up.shape)
    #print("orbital_down", orbital_down.shape)
    orb = jnp.where(spin_mask[None,:, None] == 1, orbital_up, orbital_down)
    orbitals = [orb]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, n_par)

    # Optionally apply Jastrow factor for electron cusp conditions.
    # Added pre-determinant for compatibility with pretraining.
    # Warning: this jastrow takes r_ee directly, which could violate PBC!!!
    #if jastrow_apply is not None:
    #  jastrow = jnp.exp(
    #      jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
    #  )
    #  orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply


def make_bf_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals with backflow, without jastrow and pre-orbitals.
  This only apply to Psiformer!!

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det: # this is true for Psiformer
        # Dense determinant. Need N orbitals per electron per determinant.
        #norbitals = sum(nspins) * options.determinants * num_states
        norbitals = nkpt * options.determinants * num_states
      #else:
      #  # Spin-factored block-diagonal determinant. Need nspin orbitals per
      #  # electron per determinant.
      #  orbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)
    #print("nspin_orbitals", nspin_orbitals)

    # create envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # orbital shaping
    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
              out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    key, key1 = jax.random.split(key)
    key, key2 = jax.random.split(key)
    initializer = jax.nn.initializers.lecun_normal()
    params['cmat_real'] = initializer(key1, shape=(nkpt, n_par, ndet)) + jnp.eye(nkpt, n_par)[...,None]  
    params['cmat_imag'] = initializer(key2, shape=(nkpt, n_par, ndet)) 

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    nkpt, ngpt, sdim = options.system.kp_vec.shape
    #print("nkpt",nkpt)
    n_par = options.system.n_par
    ndet = options.system.ndet

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  envelope_factor = options.envelope.apply(
    #      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    #  )
    #  h_to_orbitals = envelope_factor * h_to_orbitals

    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    #for i in range(len(h_to_orbitals)): print("h_to_orbitals i", i, h_to_orbitals[i].shape) # (n_par, n_hdim=num_heads*heads_dim)

    #active_spin_channels = [spin for spin in nspins if spin > 0]
    #active_spin_partitions = network_blocks.array_partitions(
    #    active_spin_channels
    #)

    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, nkpt*ndet*2)
    if options.complex_output:
      # create imaginary orbitals
      orbitals = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
      ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, nkpt*ndet)

    # Apply envelopes if required.
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    #  r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    #  r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    #  for i in range(len(active_spin_channels)):
    #    orbitals[i] = orbitals[i] * options.envelope.apply(
    #        ae=ae_channels[i],
    #        r_ae=r_ae_channels[i],
    #        r_ee=r_ee_channels[i],
    #        **params['envelope'][i],
    #    )

    # Reshape into matrices.
    #shapes = [
    #    (spin, -1, sum(nspins) if options.full_det else spin)
    #    for spin in active_spin_channels
    #]
    shapes = [(n_par, -1, nkpt)]
    #print("shapes",shapes)
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, ndet, nkpt)
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det: # true for Psiformer
      orbitals = [jnp.concatenate(orbitals, axis=1)]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, nkpt)

    cmat = params["cmat_real"] + 1j*params["cmat_imag"] # (nkpt, n_par, ndet)
    orbital = orbitals[0]
    orbitals_list = []
    for i in range(ndet):
      orbitals_list.append(orbital[i]@cmat[...,i])
    orbitals_list = jnp.array(orbitals_list)
    #print("orbitals_list", orbitals_list.shape)
    orbitals = [orbitals_list]
    #orbitals = [orbital@cmat for orbital in orbitals]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, n_par)

    # Optionally apply Jastrow factor for electron cusp conditions.
    # Added pre-determinant for compatibility with pretraining.
    # Warning: this jastrow takes r_ee directly, which could violate PBC!!!
    #if jastrow_apply is not None:
    #  jastrow = jnp.exp(
    #      jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
    #  )
    #  orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply


def make_wojo_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals, without jastrow and pre-orbitals.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants * num_states
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per
        # electron per determinant.
        norbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)
    #print("nspin_orbitals", nspin_orbitals)

    # create envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # orbital shaping
    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in, # num_heads*heads_dim for transformer
              out_dim=nspin_orbital, # n_par*n_det*2 for complex orbital
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  envelope_factor = options.envelope.apply(
    #      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    #  )
    #  h_to_orbitals = envelope_factor * h_to_orbitals

    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    #for i in range(len(h_to_orbitals)): print("h_to_orbitals i", i, h_to_orbitals[i].shape) # (n_par, n_hdim=num_heads*heads_dim)
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(
        active_spin_channels
    )
    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, n_par*ndet*2)
    if options.complex_output:
      # create imaginary orbitals
      orbitals = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
      ]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (n_par, n_par*ndet)

    # Apply envelopes if required.
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    #  r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    #  r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    #  for i in range(len(active_spin_channels)):
    #    orbitals[i] = orbitals[i] * options.envelope.apply(
    #        ae=ae_channels[i],
    #        r_ae=r_ae_channels[i],
    #        r_ee=r_ee_channels[i],
    #        **params['envelope'][i],
    #    )

    # Reshape into matrices.
    shapes = [
        (spin, -1, sum(nspins) if options.full_det else spin)
        for spin in active_spin_channels
    ]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det:
      orbitals = [jnp.concatenate(orbitals, axis=1)]
    #for i in range(len(orbitals)): print("orbitals i", i, orbitals[i].shape) # (ndet, n_par, n_par)

    # Optionally apply Jastrow factor for electron cusp conditions.
    # Added pre-determinant for compatibility with pretraining.
    # Warning: this jastrow takes r_ee directly, which could violate PBC!!!
    #if jastrow_apply is not None:
    #  jastrow = jnp.exp(
    #      jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
    #  )
    #  orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply


def make_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants * num_states
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per
        # electron per determinant.
        norbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)

    # create envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # orbital shaping
    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in,
              out_dim=nspin_orbital,
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      envelope_factor = options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
      )
      h_to_orbitals = envelope_factor * h_to_orbitals
    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(
        active_spin_channels
    )
    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    if options.complex_output:
      # create imaginary orbitals
      orbitals = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
      ]

    # Apply envelopes if required.
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
      r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
      r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
      for i in range(len(active_spin_channels)):
        orbitals[i] = orbitals[i] * options.envelope.apply(
            ae=ae_channels[i],
            r_ae=r_ae_channels[i],
            r_ee=r_ee_channels[i],
            **params['envelope'][i],
        )

    # Reshape into matrices.
    shapes = [
        (spin, -1, sum(nspins) if options.full_det else spin)
        for spin in active_spin_channels
    ]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det:
      orbitals = [jnp.concatenate(orbitals, axis=1)]

    # Optionally apply Jastrow factor for electron cusp conditions.
    # Added pre-determinant for compatibility with pretraining.
    # Warning: this jastrow takes r_ee directly, which could violate PBC!!!
    if jastrow_apply is not None:
      jastrow = jnp.exp(
          jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
      )
      orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply



def make_hf_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    system: Any,
    options: BaseNetworkOptions,
    #equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  #equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  #jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    #dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    active_spin_channels = [spin for spin in nspins if spin > 0] # [nup,ndown]
    #print("active_spin_channels",active_spin_channels)
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1) # equal to 1 for ground states
    #print("num_states",num_states)
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants * num_states
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per
        # electron per determinant.
        norbitals = nspin * options.determinants * num_states # (nup*ndet,ndown*ndet)
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals) # (2*nup*ndet,2*ndown*ndet)
    #print("nspin_orbitals",nspin_orbitals) 

    # create envelope params
    #natom = charges.shape[0]
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  # Applied to output from final layer of 1e stream.
    #  output_dims = dims_orbital_in
    #elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  # Applied to orbitals.
    #  if options.complex_output:
    #    output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
    #  else:
    #    output_dims = nspin_orbitals
    #else:
    #  raise ValueError('Unknown envelope type')
    #params['envelope'] = options.envelope.init(
    #    natom=natom, output_dims=output_dims, ndim=options.ndim
    #)

    ## Jastrow params.
    #if jastrow_init is not None:
    #  params['jastrow'] = jastrow_init()

    ## orbital shaping
    #orbitals = []
    #for nspin_orbital in nspin_orbitals:
    #  key, subkey = jax.random.split(key)
    #  orbitals.append(
    #      network_blocks.init_linear_layer(
    #          subkey,
    #          in_dim=dims_orbital_in,
    #          out_dim=nspin_orbital,
    #          include_bias=options.bias_orbitals,
    #      )
    #  )
    ##print("orbitals",len(orbitals)) # equal to 2 if exists both nup and ndown
    #params['orbital'] = orbitals
    
    in_dim, out_dim = system.vs_arr.shape
    key, subkey = jax.random.split(key)
    #params['uvec'] = lecun_normal(subkey, (nkpt,ngpt)) 
    params['uvec'] = 0.0*jax.random.normal(key, shape=(in_dim, out_dim)) / jnp.sqrt(float(in_dim))

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    #print("km",system.km)
    #print("kp",system.kp)
    #print("km_vec",system.km_vec.shape)
    #print("kp_vec",system.kp_vec.shape)
    #print("vs_arr",system.vs_arr.shape)
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    #print("pos",pos.shape) # (sdim*n_par)
    #print("spins",spins.shape) # (n_par)
    #print("atoms",atoms.shape) # (1,2)
    #print("ae",ae.shape) # atom-electron vector. Shape (nelectron, natom, ndim). single electron features
    #print("ee",ee.shape) # electron-electron vector. Shape (nelectron, nelectron, ndim).
    #print("r_ae",r_ae.shape) # atom-electron distance. Shape (nelectron, natom, 1).
    #print("r_ee",r_ee.shape) # electron-electron distance. Shape (nelectron, nelectron, 1).

    ############################## slaterdet orbitals ##########################################

    # just need a function that takes ae as input and output orbitals of shape [1,n_par,n_par]
    nkpt, ngpt, _ = system.kp_vec.shape
    nk_par = spins.shape[0]
    eikpr = jnp.exp(1j*jnp.dot(system.kp_vec[:nk_par]+system.kp,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    eikmr = jnp.exp(1j*jnp.dot(system.km_vec[:nk_par]+system.km,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    #tmp = jnp.exp(1j*jnp.dot(system.km_vec[:nk_par]+system.km,(pos+system.lattice.reshape(4)).reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    #jax.debug.print('{}', jnp.linalg.norm(jnp.abs(eikmr)-jnp.abs(tmp)))

    #eikpr = jnp.exp(1j*jnp.dot(system.kp_vec[:nk_par],pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    #eikmr = jnp.exp(1j*jnp.dot(system.km_vec[:nk_par],pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    vu_arr = system.vs_arr + params['uvec']
    ueikpr = jnp.sum(jnp.multiply(vu_arr[:nk_par,:ngpt][:, :, None], eikpr), axis=1)  # (nkpt,n_par)
    ueikmr = jnp.sum(jnp.multiply(vu_arr[:nk_par,ngpt:][:, :, None], eikmr), axis=1)  # (nkpt,n_par)
    #tmp2 = jnp.sum(jnp.multiply(vu_arr[:nk_par,ngpt:][:, :, None], tmp), axis=1)  # (nkpt,n_par)
    #print("eikpr",eikpr.shape)
    #print("eikmr",eikmr.shape)
    #print("ueikmr",ueikmr.shape)
    #print("ueikpr",ueikpr.shape)
    #jax.debug.print('{}', jnp.linalg.norm(jnp.abs(ueikmr)-jnp.abs(tmp2)))

    spin_mask = jnp.array((spins+1)/2, dtype=int)
    orb = jnp.where(spin_mask[None,:] == 1, ueikpr, ueikmr)
    #tmp3 = jnp.where(spin_mask[None,:] == 1, ueikpr, tmp2)
    #jax.debug.print('{}', jnp.linalg.norm(jnp.abs(orb)-jnp.abs(tmp3)))
    #print("spin_mask",spin_mask.shape)
    #print("orbitals",orb.shape)
    orb = jnp.expand_dims(orb,axis=0)
    orbitals = [orb]
    #for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (ndet,nup,nup)

    ############################### ferminet orbitals ##########################################
    #h_to_orbitals = equivariant_layers_apply(
    #    params['layers'],
    #    ae=ae,
    #    r_ae=r_ae,
    #    ee=ee,
    #    r_ee=r_ee,
    #    spins=spins,
    #    charges=charges,
    #)

    ##if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    ##  envelope_factor = options.envelope.apply(
    ##      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    ##  )
    ##  h_to_orbitals = envelope_factor * h_to_orbitals
    ## Note split creates arrays of size 0 for spin channels without electrons.
    #h_to_orbitals = jnp.split(
    #    h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    #)
    ## Drop unoccupied spin channels
    #h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0] # (nup,hdim), (ndown,hdim)
    ##for i in range(len(h_to_orbitals)): print("h_to_orbitals",h_to_orbitals[i].shape)
    #active_spin_channels = [spin for spin in nspins if spin > 0] # (nup,ndown)
    ##print("active_spin_channels",active_spin_channels)
    #active_spin_partitions = network_blocks.array_partitions(
    #    active_spin_channels
    #)
    ##print("active_spin_partitions",active_spin_partitions) # [2]
    ## Create orbitals.
    #orbitals = [
    #    network_blocks.linear_layer(h, **p)
    #    for h, p in zip(h_to_orbitals, params['orbital'])
    #]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) #  (nup,ndet*nup*2), (ndown,ndet*ndown*2) for complex
    #if options.complex_output:
    #  # create imaginary orbitals
    #  orbitals = [
    #      orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
    #  ]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) #  (nup,ndet*nup), (ndown,ndet*ndown) for complex

    ## Apply envelopes if required.
    ##if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    ##  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    ##  r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    ##  r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    ##  for i in range(len(active_spin_channels)):
    ##    orbitals[i] = orbitals[i] * options.envelope.apply(
    ##        ae=ae_channels[i],
    ##        r_ae=r_ae_channels[i],
    ##        r_ee=r_ee_channels[i],
    ##        **params['envelope'][i],
    ##    )

    ## Reshape into matrices.
    #shapes = [
    #    (spin, -1, sum(nspins) if options.full_det else spin)
    #    for spin in active_spin_channels
    #] # [(nup,-1,nup),(ndown,-1,ndown)]
    ##print("active_spin_channels",active_spin_channels) 
    ##print("nspins",nspins) 
    ##print("shapes",shapes)
    #orbitals = [
    #    jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    #]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (nup,ndet,nup), (ndown,ndet,,ndown)
    #orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (ndet,nup,nup),(ndet,ndown,ndown)
    ##if options.full_det:
    ##  orbitals = [jnp.concatenate(orbitals, axis=1)]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape)

    ## Optionally apply Jastrow factor for electron cusp conditions.
    ## Added pre-determinant for compatibility with pretraining.
    ##if jastrow_apply is not None:
    ##  jastrow = jnp.exp(
    ##      jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
    ##  )
    ##  orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply


def make_graph_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    system: Any,
    options: BaseNetworkOptions,
    #equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  #equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  #jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    #dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    active_spin_channels = [spin for spin in nspins if spin > 0] # [nup,ndown]
    #print("active_spin_channels",active_spin_channels)
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1) # equal to 1 for ground states
    #print("num_states",num_states)
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants * num_states
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per
        # electron per determinant.
        norbitals = nspin * options.determinants * num_states # (nup*ndet,ndown*ndet)
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals) # (2*nup*ndet,2*ndown*ndet)
    #print("nspin_orbitals",nspin_orbitals) 

    # create envelope params
    #natom = charges.shape[0]
    #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    #  # Applied to output from final layer of 1e stream.
    #  output_dims = dims_orbital_in
    #elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    #  # Applied to orbitals.
    #  if options.complex_output:
    #    output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
    #  else:
    #    output_dims = nspin_orbitals
    #else:
    #  raise ValueError('Unknown envelope type')
    #params['envelope'] = options.envelope.init(
    #    natom=natom, output_dims=output_dims, ndim=options.ndim
    #)

    ## Jastrow params.
    #if jastrow_init is not None:
    #  params['jastrow'] = jastrow_init()

    ## orbital shaping
    #orbitals = []
    #for nspin_orbital in nspin_orbitals:
    #  key, subkey = jax.random.split(key)
    #  orbitals.append(
    #      network_blocks.init_linear_layer(
    #          subkey,
    #          in_dim=dims_orbital_in,
    #          out_dim=nspin_orbital,
    #          include_bias=options.bias_orbitals,
    #      )
    #  )
    ##print("orbitals",len(orbitals)) # equal to 2 if exists both nup and ndown
    #params['orbital'] = orbitals
    
    in_dim, out_dim = system.vs_arr.shape
    key, subkey = jax.random.split(key)
    #params['uvec'] = lecun_normal(subkey, (nkpt,ngpt)) 
    params['uvec'] = 0.0*jax.random.normal(key, shape=(in_dim, out_dim)) / jnp.sqrt(float(in_dim))

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    let me sketch the network:
      graph nn output gi, bf 
````  phi_k(x) = \sum u_kg exp(i(k+g)(x+bf))
      phi_y(x) = \sum_k cmat_yk phi_k(x) * orbs_k(x)
      orbs_k(x) = orbs(gi,k) 

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """

    #print("km",system.km)
    #print("kp",system.kp)
    #print("km_vec",system.km_vec.shape)
    #print("kp_vec",system.kp_vec.shape)
    #print("vs_arr",system.vs_arr.shape)
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    #print("pos",pos.shape) # (sdim*n_par)
    #print("spins",spins.shape) # (n_par)
    #print("atoms",atoms.shape) # (1,2)
    #print("ae",ae.shape) # atom-electron vector. Shape (nelectron, natom, ndim). single electron features
    #print("ee",ee.shape) # electron-electron vector. Shape (nelectron, nelectron, ndim).
    #print("r_ae",r_ae.shape) # atom-electron distance. Shape (nelectron, natom, 1).
    #print("r_ee",r_ee.shape) # electron-electron distance. Shape (nelectron, nelectron, 1).

    ############################## slaterdet orbitals ##########################################

    # just need a function that takes ae as input and output orbitals of shape [1,n_par,n_par]
    nkpt, ngpt, _ = system.kp_vec.shape
    nk_par = spins.shape[0]
    eikpr = jnp.exp(1j*jnp.dot(system.kp_vec[:nk_par]+system.kp,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    eikmr = jnp.exp(1j*jnp.dot(system.km_vec[:nk_par]+system.km,pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    #eikpr = jnp.exp(1j*jnp.dot(system.kp_vec[:nk_par],pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    #eikmr = jnp.exp(1j*jnp.dot(system.km_vec[:nk_par],pos.reshape(-1,2).T)) # (nkpt,ngpt,n_par)
    vu_arr = system.vs_arr + params['uvec']
    ueikpr = jnp.sum(jnp.multiply(vu_arr[:nk_par,:ngpt][:, :, None], eikpr), axis=1)  # (nkpt,n_par)
    ueikmr = jnp.sum(jnp.multiply(vu_arr[:nk_par,ngpt:][:, :, None], eikmr), axis=1)  # (nkpt,n_par)
    #print("eikpr",eikpr.shape)
    #print("eikmr",eikmr.shape)
    #print("ueikmr",ueikmr.shape)
    #print("ueikpr",ueikpr.shape)

    spin_mask = jnp.array((spins+1)/2, dtype=int)
    orb = jnp.where(spin_mask[None,:] == 1, ueikpr, ueikmr)
    #print("spin_mask",spin_mask.shape)
    #print("orbitals",orb.shape)
    orb = jnp.expand_dims(orb,axis=0)
    orbitals = [orb]
    #for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (ndet,nup,nup)

    ############################### ferminet orbitals ##########################################
    #h_to_orbitals = equivariant_layers_apply(
    #    params['layers'],
    #    ae=ae,
    #    r_ae=r_ae,
    #    ee=ee,
    #    r_ee=r_ee,
    #    spins=spins,
    #    charges=charges,
    #)

    ##if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    ##  envelope_factor = options.envelope.apply(
    ##      ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
    ##  )
    ##  h_to_orbitals = envelope_factor * h_to_orbitals
    ## Note split creates arrays of size 0 for spin channels without electrons.
    #h_to_orbitals = jnp.split(
    #    h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    #)
    ## Drop unoccupied spin channels
    #h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0] # (nup,hdim), (ndown,hdim)
    ##for i in range(len(h_to_orbitals)): print("h_to_orbitals",h_to_orbitals[i].shape)
    #active_spin_channels = [spin for spin in nspins if spin > 0] # (nup,ndown)
    ##print("active_spin_channels",active_spin_channels)
    #active_spin_partitions = network_blocks.array_partitions(
    #    active_spin_channels
    #)
    ##print("active_spin_partitions",active_spin_partitions) # [2]
    ## Create orbitals.
    #orbitals = [
    #    network_blocks.linear_layer(h, **p)
    #    for h, p in zip(h_to_orbitals, params['orbital'])
    #]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) #  (nup,ndet*nup*2), (ndown,ndet*ndown*2) for complex
    #if options.complex_output:
    #  # create imaginary orbitals
    #  orbitals = [
    #      orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
    #  ]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) #  (nup,ndet*nup), (ndown,ndet*ndown) for complex

    ## Apply envelopes if required.
    ##if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    ##  ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    ##  r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    ##  r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    ##  for i in range(len(active_spin_channels)):
    ##    orbitals[i] = orbitals[i] * options.envelope.apply(
    ##        ae=ae_channels[i],
    ##        r_ae=r_ae_channels[i],
    ##        r_ee=r_ee_channels[i],
    ##        **params['envelope'][i],
    ##    )

    ## Reshape into matrices.
    #shapes = [
    #    (spin, -1, sum(nspins) if options.full_det else spin)
    #    for spin in active_spin_channels
    #] # [(nup,-1,nup),(ndown,-1,ndown)]
    ##print("active_spin_channels",active_spin_channels) 
    ##print("nspins",nspins) 
    ##print("shapes",shapes)
    #orbitals = [
    #    jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    #]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (nup,ndet,nup), (ndown,ndet,,ndown)
    #orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape) # (ndet,nup,nup),(ndet,ndown,ndown)
    ##if options.full_det:
    ##  orbitals = [jnp.concatenate(orbitals, axis=1)]
    ##for i in range(len(orbitals)): print("orbitals i", orbitals[i].shape)

    ## Optionally apply Jastrow factor for electron cusp conditions.
    ## Added pre-determinant for compatibility with pretraining.
    ##if jastrow_apply is not None:
    ##  jastrow = jnp.exp(
    ##      jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
    ##  )
    ##  orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply

def make_laughlin_orbitals(
        nspins: tuple[int, int],
        charges: jnp.ndarray,
        system: any,
        options: any
) -> tuple[callable, callable]:
    def init(key: chex.PRNGKey) -> dict:
        """Returns initial parameters for Laughlin orbitals."""
        params = {}
        # Laughlin exponent - typically odd integers (m=3 for ¦Í=1/3 filling)
        params['m'] = jnp.array(system.m if hasattr(system, 'm') else 3.0)
        # Magnetic length parameter
        params['magnetic_length'] = jnp.array(
            system.magnetic_length if hasattr(system, 'magnetic_length') else 1.0)
        # Optional scaling for the gaussian confinement
        params['confinement_scale'] = jnp.array(
            system.confinement_scale if hasattr(system, 'confinement_scale') else 0.25)
        return params

    def apply(
            params,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> tuple[jnp.ndarray]:
    
        dtype = jnp.complex128

        m = params['m']
        magnetic_length = params['magnetic_length']
        confinement_scale = params['confinement_scale']

        # Convert positions to complex coordinates
        n_electrons = spins.shape[0]
        z = (pos.reshape(-1, 2)[:, 0] + 1j * pos.reshape(-1, 2)[:, 1]).astype(dtype)

        # Define F(z) function in log domain - typically a Gaussian factor
        def log_F(z_val):
            # For the lowest Landau level: F(z) = exp(-|z|2/(4l2))
            # log(F(z)) = -|z|2/(4l2)
            return -jnp.abs(z_val) ** 2 / (4.0 * magnetic_length ** 2)

        # Create the matrix for the determinant in complex form
        # We'll compute the constituent parts in log form when possible
        
        # Compute log_F values for all z positions at once
        log_F_values = -jnp.abs(z) ** 2 / (4.0 * magnetic_length ** 2* confinement_scale)

        def build_matrix_element(i, j):
            log_f_zi = log_F_values[i]
    
        # Use a direct function to handle the j==0 case
            log_mag = jnp.where(
              j == 0,
              log_f_zi,                                      # j=0 case
              j * jnp.log(jnp.abs(z[i]) + 1e-10) + log_f_zi  # j>0 case
            )
    
            phase = jnp.where(
              j == 0,
              0.0,                  # j=0 case
              j * jnp.angle(z[i])   # j>0 case
            )
    
            return jnp.exp(log_mag + 1j * phase)


        row_fn = jax.vmap(build_matrix_element, in_axes=(0, None))
        matrix_fn = jax.vmap(row_fn, in_axes=(None, 0))
        matrix = matrix_fn(jnp.arange(n_electrons), jnp.arange(n_electrons))

        # Handle Jastrow factor 
        def apply_jastrow_factor(matrix, m):
        # Only apply if m > 1
            def jastrow_fn(m_val):
        # Get indices for all pairs i<j
              i_indices, j_indices = jnp.triu_indices(n_electrons, k=1)
        
        # Get z values for each pair
              zi = z[i_indices]
              zj = z[j_indices]
              z_diff = zi - zj
        
        # Add a small value to z_diff where it might be close to zero
              epsilon = 1e-10
              safe_z_diff = jnp.where(jnp.abs(z_diff) < epsilon, z_diff + epsilon, z_diff)
        
        # Compute log of pair contributions and sum them
              pair_logs = (m_val - 1) * jnp.log(safe_z_diff).astype(dtype)
              jastrow_log = jnp.sum(pair_logs)
        
        # Apply the Jastrow factor evenly across all matrix elements
              jastrow_per_element = jnp.exp(jastrow_log / n_electrons)
              return matrix * jastrow_per_element
    
        # Only apply Jastrow if m > 1, otherwise return matrix unchanged
            return jax.lax.cond(m > 1, jastrow_fn, lambda _: matrix, m)

        matrix = apply_jastrow_factor(matrix, m)

        # Reshape the matrix for the framework's determinant calculation
        matrix = matrix.reshape(1, n_electrons, n_electrons)

        return (matrix,)

    return init, apply

def make_bcs_orbitals(
        nspins: Tuple[int, int],
        charges: jnp.ndarray,
        system: Any,
        options: BaseNetworkOptions,
) -> Tuple[callable, callable]:

    def init(key: chex.PRNGKey) -> ParamTree:

        key, subkey = jax.random.split(key)
        params = {}

        # Initialize pairing gap parameter (can be fixed or learned)
        params['delta'] = jnp.array(system.delta if hasattr(system, 'delta') else 1.0)

        # Initialize chemical potential (can be fixed or learned)
        params['mu'] = jnp.array(system.mu if hasattr(system, 'mu') else 1.0)

        # Momentum cutoff scale (typically fixed as a multiple of sqrt(mu))
        # params['k_c'] = jnp.array(system.k_c if hasattr(system, 'k_c') else 2.0 * jnp.sqrt(params['mu']))

        # Optional: learnable perturbation to the pairing function
        if hasattr(system, 'use_perturbation') and system.use_perturbation:
            key, subkey = jax.random.split(key)
            # Small initial perturbation
            params['phi_perturb'] = 0.01 * jax.random.normal(
                subkey, shape=(options.n_grid, options.n_grid) if hasattr(options, 'n_grid') else (10, 10)
            )

        return params  # Return initialized parameters for the BCS orbital

    def apply(
            params: ParamTree,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> Sequence[jnp.ndarray]:
    
        delta = params['delta']
        mu = params['mu']
        k_c = 2.0

        # Reshape positions to get positions for each electron
        n_electrons = spins.shape[0]  # Total number of electrons
        positions = pos.reshape(n_electrons, options.ndim)
        
        n_up = nspins[0]
        n_down = nspins[1]
        
        up_indices = jnp.array([i for i in range(n_electrons) if i < n_up])
        down_indices = jnp.array([i for i in range(n_electrons) if i >= n_up])
        positions_up = jnp.take(positions, up_indices, axis=0)
        positions_down = jnp.take(positions, down_indices, axis=0)

        # Build pairing matrix (each entry is phi_r for pairs of up and down electrons)
        n_up = positions_up.shape[0]
        n_down = positions_down.shape[0]

        # The matrix must be square for the determinant calculation
        if n_up != n_down:
            raise ValueError(f"BCS wavefunction requires equal numbers of spin-up and spin-down electrons, "
                             f"but got {n_up} up and {n_down} down.")

        # Compute relative positions between all pairs
        rel_positions = jnp.zeros((n_up, n_down, options.ndim))
        for i in range(n_up):
            for j in range(n_down):
                rel_positions = rel_positions.at[i, j].set(positions_up[i] - positions_down[j])
        def phi_r(r_vec, delta, mu, k_c, Lx, Ly):
            norm_factor = 1.0 / (Lx * Ly)
    
    # Use a fixed size for the k-space grid that's sufficiently large
    # This avoids dynamic shape errors in JAX
            max_n = 50  # Choose this to be larger than any realistic k_c * L / (2¦Ð)
    
    # Create fixed-size ranges for k-space sampling
            nx_range = jnp.arange(-max_n, max_n + 1)
            ny_range = jnp.arange(-max_n, max_n + 1)
    
    # Create meshgrid with fixed dimensions
            nx_grid, ny_grid = jnp.meshgrid(nx_range, ny_range)
            nx_flat = nx_grid.flatten()
            ny_flat = ny_grid.flatten()
    
    # Compute kx, ky values
            kx = 2 * jnp.pi * nx_flat / Lx
            ky = 2 * jnp.pi * ny_flat / Ly
            k_squared = kx**2 + ky**2
    
    # BCS coherence factors
            xi_k = k_squared / 2 - mu
            E_k = jnp.sqrt(xi_k**2 + delta**2)
            denominator = xi_k + E_k
    
    # Phase factors
            k_dot_r = kx * r_vec[0] + ky * r_vec[1]
            phase = jnp.exp(1j * k_dot_r)
    
    # Apply cutoff and avoid division by zero
            in_cutoff = k_squared <= k_c**2
            valid_denominator = jnp.abs(denominator) >= 1e-10
            mask = in_cutoff & valid_denominator
    
    # Calculate contribution for each k point
            integrand = jnp.where(
                mask,
                delta / denominator * phase,
                jnp.zeros_like(denominator, dtype=jnp.complex64)
            )
    
    # Sum all contributions
            result = jnp.sum(integrand)
    
            return norm_factor * result

        # For the loops calculating rel_positions and M:
        def compute_rel_positions(positions_up, positions_down, Lx, Ly):
            n_up = positions_up.shape[0]
            n_down = positions_down.shape[0]
    
    # Vectorized computation for all pairs
    # This avoids dynamic indexing that can cause tracing errors
            positions_up_expanded = jnp.expand_dims(positions_up, 1)  # shape [n_up, 1, 2]
            positions_down_expanded = jnp.expand_dims(positions_down, 0)  # shape [1, n_down, 2]
    
    # Compute raw displacements
            raw_displacements = positions_up_expanded - positions_down_expanded  # shape [n_up, n_down, 2]
    
    # Apply periodic boundary conditions
            dx = raw_displacements[..., 0] - Lx * jnp.round(raw_displacements[..., 0] / Lx)
            dy = raw_displacements[..., 1] - Ly * jnp.round(raw_displacements[..., 1] / Ly)
    
    # Stack back into [n_up, n_down, 2] shape
            displacements = jnp.stack([dx, dy], axis=-1)
    
            return displacements
        
        def compute_pairing_matrix(rel_positions, delta, mu, k_c, Lx, Ly):
            n_up, n_down = rel_positions.shape[0], rel_positions.shape[1]
    
    # Using scan instead of a direct vmap on phi_r to avoid tracing issues
            def phi_for_one_pair(i, j):
                return phi_r(rel_positions[i, j], delta, mu, k_c, Lx, Ly)
    
    # Initialize result matrix
            M = jnp.zeros((n_up, n_down), dtype=jnp.complex64)
    
    # Loop over positions using a simpler approach for JAX
    # We use a scan-like implementation with explicit indexing
            for i in range(n_up):
                for j in range(n_down):
                    M = M.at[i, j].set(phi_r(rel_positions[i, j], delta, mu, k_c, Lx, Ly))
    
            return M

        if hasattr(system, 'slattice'):
            lattice = system.slattice

            Lx = jnp.sqrt(jnp.sum(lattice[0] ** 2))
            Ly = jnp.sqrt(jnp.sum(lattice[1] ** 2))
        else:

            #Lx = Ly = params.get('box_size', 10.0)
            Lx = Ly =10.0
            
        rel_positions = compute_rel_positions(positions_up, positions_down, Lx, Ly)
        M = compute_pairing_matrix(rel_positions, delta, mu, k_c, Lx, Ly)

        return [M]

    return init, apply


## Excited States  ##


def make_state_matrix(signed_network: FermiNetLike, n: int) -> FermiNetLike:
  """Construct a matrix-output ansatz which gives the Slater matrix of states.

  Let signed_network(params, pos, spins, options) be a function which returns
  psi_1(pos), psi_2(pos), ... psi_n(pos) as a pair of arrays of length n, one
  with values of sign(psi_k), one with values of log(psi_k). Then this function
  returns a new function which computes the matrix psi_i(pos_j), given an array
  of positions (and possibly spins) which has n times as many dimensions as
  expected by signed_network. The output of this new meta-matrix is also given
  as a sign, log pair.

  Args:
    signed_network: A function with the same calling convention as the FermiNet.
    n: the number of excited states, needed to know how to shape the determinant

  Returns:
    A function with two outputs which combines the individual excited states
    into a matrix of wavefunctions, one with the sign and one with the log.
  """

  def state_matrix(params, pos, spins, atoms, charges):
    """Evaluate state_matrix for a given ansatz."""
    # `pos` has shape (n*nelectron*ndim), but can be reshaped as
    # (n, nelectron, ndim), that is, the first dimension indexes which excited
    # state we are considering, the second indexes electrons, and the third
    # indexes spatial dimensions. `spins` has the same ordering of indices,
    # but does not have the spatial dimensions. `atoms` does not have the
    # leading index of number of excited states, as the different states are
    # always evaluated at the same atomic geometry.
    pos_ = jnp.reshape(pos, [n, -1])
    spins_ = jnp.reshape(spins, [n, -1])
    vmap_network = jax.vmap(signed_network, (None, 0, 0, None, None))
    sign_mat, log_mat = vmap_network(params, pos_, spins_, atoms, charges)
    return sign_mat, log_mat

  return state_matrix


def make_state_trace(signed_network: FermiNetLike, n: int) -> FermiNetLike:
  """Construct a single-output f'n which gives the trace over the state matrix.

  Returns the sum of the diagonal of the matrix of log|psi| values created by
  make_state_matrix. That means for a set of inputs x_1, ..., x_n, instead of
  returning the full matrix of psi_i(x_j), only return the sum of the diagonal
  sum_i log(psi_i(x_i)), so one state per input. Used for MCMC sampling.

  Args:
    signed_network: A function with the same calling convention as the FermiNet.
    n: the number of excited states, needed to know how to shape the determinant

  Returns:
    A function with a multiple outputs which takes a set of inputs and returns
    one output per input.
  """
  state_matrix = make_state_matrix(signed_network, n)

  def state_trace(params, pos, spins, atoms, charges, **kwargs):
    """Evaluate trace of the state matrix for a given ansatz."""
    _, log_in = state_matrix(
        params, pos, spins, atoms=atoms, charges=charges, **kwargs)

    return jnp.trace(log_in)

  return state_trace


def make_total_ansatz(signed_network: FermiNetLike,
                      n: int,
                      complex_output: bool = False) -> FermiNetLike:
  """Construct a single-output ansatz which gives the meta-Slater determinant.

  Let signed_network(params, pos, spins, options) be a function which returns
  psi_1(pos), psi_2(pos), ... psi_n(pos) as a pair of arrays, one with values
  of sign(psi_k), one with values of log(psi_k). Then this function returns a
  new function which computes det[psi_i(pos_j)], given an array of positions
  (and possibly spins) which has n times as many dimensions as expected by
  signed_network. The output of this new meta-determinant is also given as a
  sign, log pair.

  Args:
    signed_network: A function with the same calling convention as the FermiNet.
    n: the number of excited states, needed to know how to shape the determinant
    complex_output: If true, the output of the network is complex, and the
      individual states return phase angles rather than signs.

  Returns:
    A function with a single output which combines the individual excited states
    into a greater wavefunction given by the meta-Slater determinant.
  """
  state_matrix = make_state_matrix(signed_network, n)

  def total_ansatz(params, pos, spins, atoms, charges, **kwargs):
    """Evaluate meta_determinant for a given ansatz."""
    sign_in, log_in = state_matrix(
        params, pos, spins, atoms=atoms, charges=charges, **kwargs)

    logmax = jnp.max(log_in)  # logsumexp trick
    if complex_output:
      # sign_in is a phase angle rather than a sign for complex networks
      mat_in = jnp.exp(log_in + 1.j * sign_in - logmax)
      sign_out, log_out = jnp.linalg.slogdet(mat_in)
      sign_out = jnp.angle(sign_out)
    else:
      sign_out, log_out = jnp.linalg.slogdet(sign_in * jnp.exp(log_in - logmax))
    log_out += n * logmax
    return sign_out, log_out

  return total_ansatz


## FermiNet ##


def make_fermi_net(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    system: Any,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.NONE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    full_det: bool = True,
    rescale_inputs: bool = False,
    # FermiNet-specific kwargs below.
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    use_last_layer: bool = False,
    separate_spin_channels: bool = False,
    schnet_electron_electron_convolutions: Tuple[int, ...] = tuple(),
    electron_nuclear_aux_dims: Tuple[int, ...] = tuple(),
    nuclear_embedding_dim: int = 0,
    schnet_electron_nuclear_convolutions: Tuple[int, ...] = tuple(),
) -> Network:
  """Creates functions for initializing parameters and evaluating ferminet.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or no jastrow if 'default'.
    complex_output: If true, the network outputs complex numbers.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    separate_spin_channels: Use separate learnable parameters for pairs of
      spin-parallel and spin-antiparallel electrons.
    schnet_electron_electron_convolutions: Dimension of embeddings used for
      electron-electron SchNet-style convolutions.
    electron_nuclear_aux_dims: hidden units in each layer of the
      electron-nuclear auxiliary stream. Used in electron-nuclear SchNet-style
      convolutions.
    nuclear_embedding_dim: Dimension of embedding used in for the nuclear
      features. Used in electron-nuclear SchNet-style convolutions.
    schnet_electron_nuclear_convolutions: Dimension of embeddings used for
      electron-nuclear SchNet-style convolutions.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network. If
    options.states > 1, the length of the vectors returned by apply are equal
    to the number of states.
  """
  if sum([nspin for nspin in nspins if nspin > 0]) == 0:
    raise ValueError('No electrons present!')

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.NONE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = FermiNetOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      rescale_inputs=rescale_inputs,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=full_det,
      hidden_dims=hidden_dims,
      separate_spin_channels=separate_spin_channels,
      schnet_electron_electron_convolutions=schnet_electron_electron_convolutions,
      electron_nuclear_aux_dims=electron_nuclear_aux_dims,
      nuclear_embedding_dim=nuclear_embedding_dim,
      schnet_electron_nuclear_convolutions=schnet_electron_nuclear_convolutions,
      use_last_layer=use_last_layer,
  )

  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    if options.bias_orbitals:
      raise ValueError('Cannot bias orbitals w/STO envelope.')

  equivariant_layers = make_fermi_net_layers(nspins, charges.shape[0], options)

  #orbitals_init, orbitals_apply = make_orbitals(
  orbitals_init, orbitals_apply = make_wojo_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=equivariant_layers,
  )

  def init(key: chex.PRNGKey) -> ParamTree:
    key, subkey = jax.random.split(key, num=2)
    return orbitals_init(subkey)

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network for a single datum.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute of the network evaluated at x.
    """
    #print("pos",pos.shape) # (sdim*n_par,)
    #print("spins",spins.shape) # (n_par,)

    orbitals = orbitals_apply(params, pos, spins, atoms, charges) # list of 2 for spin up and down, each list is of shape (ndet,nup,nup), (ndet,ndown,ndown)
    #for i in range(len(orbitals)): print("orbitals i",i,orbitals[i].shape)
    #if options.states:
    #  batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
    #  orbitals = [
    #      jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
    #      for orbital in orbitals
    #  ]
    #  result = batch_logdet_matmul(orbitals)
    #else:
    #  result = network_blocks.logdet_matmul(orbitals) # this is for state=0
    result = network_blocks.logdet_matmul(orbitals) # this is for state=0; it returns phase_out, log_out

    #if 'state_scale' in params:
    #  # only used at inference time for excited states
    #  result = result[0], result[1] + params['state_scale']
    return result

  def apply_sym(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    #pre_result = apply(params, pos, spins, atoms, charges)
    #phase_out, log_out  = pre_result

    logdet_list = []
    phase_list = []
    for i in range(system.rs):
      for j in range(system.rs):
        #print(pos.shape)
        #print(self.lattice[0].shape)
        #print(self.lattice[1].shape)
        shift = i*system.lattice[0]+j*system.lattice[1]
        pos_tmp = (pos.reshape(system.n_par,2) + shift).reshape(-1)
        phase_tmp, log_tmp = apply(params, pos_tmp, spins, atoms, charges)
        phase_list.append(phase_tmp - system.sym_vec@shift)
        logdet_list.append(log_tmp)
    phase_arr = jnp.exp(1j*jnp.array(phase_list)) # (ndet,nbatch)
    logdet_arr = jnp.array(logdet_list) # (ndet,nbatch)
    maxlogdet = jnp.max(logdet_arr,axis=0) # (nbatch,)
    #print('phase_arr', phase_arr.shape) 
    #print('logdet_arr', logdet_arr.shape) 
    #print('maxlogdet', maxlogdet.shape) 
    det = phase_arr * jnp.exp(logdet_arr - maxlogdet) # (ndet,nbatch)
    #print('det', det.shape)
    #result = jnp.matmul(wd, det) # (nbatch)
    result = jnp.sum(det,axis=0) # (nbatch)
    phase_out = jnp.angle(result) # (nbatch)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet # (nbatch)
    #print('result', result.shape)
    #print('phase_out', phase_out.shape)
    #print('log_out', log_out.shape)
    #psi = log_out + 1j*phase_out #jnp.log(phase_out.astype(complex)) # (nbatch)
    #print('psi', psi.shape)

    return phase_out, log_out 

  return Network(
      #options=options, init=init, apply=apply, orbitals=orbitals_apply
      options=options, init=init, apply=apply, apply_sym=apply_sym, orbitals=orbitals_apply
  )


def make_hf_net(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    system: Any,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    #envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    #jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.NONE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    full_det: bool = True,
    rescale_inputs: bool = False,
    # FermiNet-specific kwargs below.
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    use_last_layer: bool = False,
    separate_spin_channels: bool = False,
    schnet_electron_electron_convolutions: Tuple[int, ...] = tuple(),
    electron_nuclear_aux_dims: Tuple[int, ...] = tuple(),
    nuclear_embedding_dim: int = 0,
    schnet_electron_nuclear_convolutions: Tuple[int, ...] = tuple(),
) -> Network:
  """Creates functions for initializing parameters and evaluating hartree fock solution.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or no jastrow if 'default'.
    complex_output: If true, the network outputs complex numbers.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    separate_spin_channels: Use separate learnable parameters for pairs of
      spin-parallel and spin-antiparallel electrons.
    schnet_electron_electron_convolutions: Dimension of embeddings used for
      electron-electron SchNet-style convolutions.
    electron_nuclear_aux_dims: hidden units in each layer of the
      electron-nuclear auxiliary stream. Used in electron-nuclear SchNet-style
      convolutions.
    nuclear_embedding_dim: Dimension of embedding used in for the nuclear
      features. Used in electron-nuclear SchNet-style convolutions.
    schnet_electron_nuclear_convolutions: Dimension of embeddings used for
      electron-nuclear SchNet-style convolutions.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network. If
    options.states > 1, the length of the vectors returned by apply are equal
    to the number of states.
  """
  #print("km",system.km)
  #print("kp",system.kp)
  if sum([nspin for nspin in nspins if nspin > 0]) == 0:
    raise ValueError('No electrons present!')

  #if not envelope:
  #  envelope = envelopes.make_isotropic_envelope()

  #if not feature_layer:
  #  natoms = charges.shape[0]
  #  feature_layer = make_ferminet_features(
  #      natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
  #  )

  #if isinstance(jastrow, str):
  #  if jastrow.upper() == 'DEFAULT':
  #    jastrow = jastrows.JastrowType.NONE
  #  else:
  #    jastrow = jastrows.JastrowType[jastrow.upper()]


  options = FermiNetOptions(
      ndim=ndim,
      system=system,
      determinants=determinants,
      states=states,
      rescale_inputs=rescale_inputs,
      #envelope=envelope,
      feature_layer=feature_layer,
      #jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=full_det,
      hidden_dims=hidden_dims,
      separate_spin_channels=separate_spin_channels,
      schnet_electron_electron_convolutions=schnet_electron_electron_convolutions,
      electron_nuclear_aux_dims=electron_nuclear_aux_dims,
      nuclear_embedding_dim=nuclear_embedding_dim,
      schnet_electron_nuclear_convolutions=schnet_electron_nuclear_convolutions,
      use_last_layer=use_last_layer,
  )

  #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
  #  if options.bias_orbitals:
  #    raise ValueError('Cannot bias orbitals w/STO envelope.')

  #equivariant_layers = make_fermi_net_layers(nspins, charges.shape[0], options) # options contain pbc feature_layer

  #orbitals_init, orbitals_apply = make_orbitals(
  orbitals_init, orbitals_apply = make_hf_orbitals(
      nspins=nspins,
      charges=charges,
      system=system,
      options=options, # options contain pbc feature_layer
      #equivariant_layers=equivariant_layers,
  )

  def init(key: chex.PRNGKey) -> ParamTree:
    key, subkey = jax.random.split(key, num=2)
    return orbitals_init(subkey)

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network for a single datum.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute of the network evaluated at x.
    """
    #print("pos",pos.shape) # (sdim*n_par,)
    #print("spins",spins.shape) # (n_par,)

    #print("determinants",determinants)
    orbitals = orbitals_apply(params, pos, spins, atoms, charges) # list of 2 for spin up and down, each list is of shape (ndet,nup,nup), (ndet,ndown,ndown)
    #tmp = orbitals_apply(params, pos+system.lattice.reshape(-1,4), spins, atoms, charges) # list of 2 for spin up and down, each list is of shape (ndet,nup,nup), (ndet,ndown,ndown)
    #for i in range(len(orbitals)): print("orbitals i",i,orbitals[i].shape)
    #if options.states:
    #  batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
    #  orbitals = [
    #      jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
    #      for orbital in orbitals
    #  ]
    #  result = batch_logdet_matmul(orbitals)
    #else:
    #  result = network_blocks.logdet_matmul(orbitals) # this is for state=0
    result = network_blocks.logdet_matmul(orbitals) # this is for state=0; it returns phase_out, log_out
    #tmp2 = network_blocks.logdet_matmul(tmp) # this is for state=0; it returns phase_out, log_out
    #jax.debug.print("{}", jnp.linalg.norm(result[1]-tmp2[1]))
    #jax.debug.print("{}", jnp.linalg.norm(result[0]-tmp2[0]))

    #if 'state_scale' in params:
    #  # only used at inference time for excited states
    #  result = result[0], result[1] + params['state_scale']
    return result

  def apply_sym(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:

    #pre_result = apply(params, pos, spins, atoms, charges)
    #phase_out, log_out  = pre_result

    logdet_list = []
    phase_list = []
    for i in range(system.rs):
      for j in range(system.rs):
        #print(pos.shape)
        #print(self.lattice[0].shape)
        #print(self.lattice[1].shape)
        shift = i*system.lattice[0]+j*system.lattice[1]
        pos_tmp = (pos.reshape(system.n_par,2) + shift).reshape(-1)
        phase_tmp, log_tmp = apply(params, pos_tmp, spins, atoms, charges)
        phase_list.append(phase_tmp - system.sym_vec@shift)
        logdet_list.append(log_tmp)
    phase_arr = jnp.exp(1j*jnp.array(phase_list)) # (ndet,nbatch)
    logdet_arr = jnp.array(logdet_list) # (ndet,nbatch)
    maxlogdet = jnp.max(logdet_arr,axis=0) # (nbatch,)
    #print('phase_arr', phase_arr.shape) 
    #print('logdet_arr', logdet_arr.shape) 
    #print('maxlogdet', maxlogdet.shape) 
    det = phase_arr * jnp.exp(logdet_arr - maxlogdet) # (ndet,nbatch)
    #print('det', det.shape)
    #result = jnp.matmul(wd, det) # (nbatch)
    result = jnp.sum(det,axis=0) # (nbatch)
    phase_out = jnp.angle(result) # (nbatch)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet # (nbatch)
    #print('result', result.shape)
    #print('phase_out', phase_out.shape)
    #print('log_out', log_out.shape)
    #psi = log_out + 1j*phase_out #jnp.log(phase_out.astype(complex)) # (nbatch)
    #print('psi', psi.shape)

    return phase_out, log_out 

  def apply_osym(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """orbital symmetrization."""

    #pre_result = apply(params, pos, spins, atoms, charges)
    #phase_out, log_out  = pre_result

    orbitals_list = []
    for i in range(system.rs):
      for j in range(system.rs):
        #print(pos.shape)
        #print(self.lattice[0].shape)
        #print(self.lattice[1].shape)
        shift = i*system.lattice[0]+j*system.lattice[1]
        pos_tmp = (pos.reshape(system.n_par,2) + shift).reshape(-1)
        #print(i,j,pos_tmp.shape)
        orbitals_tmp = orbitals_apply(params, pos_tmp, spins, atoms, charges)[0]
        #print("orbitals_tmp", orbitals_tmp.shape)
        orbitals_list.append(orbitals_tmp*jnp.exp(-1j*system.sym_vec@shift))

    orbitals = jnp.array(orbitals_list)
    #print("orbitals", orbitals.shape)
    orbitals = jnp.sum(orbitals,axis=0) # (nbatch)
    #print("orbitals", orbitals.shape)
    result = network_blocks.logdet_matmul(orbitals)

    return result

  return Network(
      #options=options, init=init, apply=apply, orbitals=orbitals_apply
      options=options, init=init, apply=apply, apply_sym=apply_sym, apply_osym=apply_osym, orbitals=orbitals_apply
  )


def make_graph_net(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    system: Any,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    #envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    #jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.NONE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    full_det: bool = True,
    rescale_inputs: bool = False,
    # FermiNet-specific kwargs below.
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    use_last_layer: bool = False,
    separate_spin_channels: bool = False,
    schnet_electron_electron_convolutions: Tuple[int, ...] = tuple(),
    electron_nuclear_aux_dims: Tuple[int, ...] = tuple(),
    nuclear_embedding_dim: int = 0,
    schnet_electron_nuclear_convolutions: Tuple[int, ...] = tuple(),
) -> Network:
  """Creates functions for initializing parameters and evaluating hartree fock solution.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or no jastrow if 'default'.
    complex_output: If true, the network outputs complex numbers.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    separate_spin_channels: Use separate learnable parameters for pairs of
      spin-parallel and spin-antiparallel electrons.
    schnet_electron_electron_convolutions: Dimension of embeddings used for
      electron-electron SchNet-style convolutions.
    electron_nuclear_aux_dims: hidden units in each layer of the
      electron-nuclear auxiliary stream. Used in electron-nuclear SchNet-style
      convolutions.
    nuclear_embedding_dim: Dimension of embedding used in for the nuclear
      features. Used in electron-nuclear SchNet-style convolutions.
    schnet_electron_nuclear_convolutions: Dimension of embeddings used for
      electron-nuclear SchNet-style convolutions.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network. If
    options.states > 1, the length of the vectors returned by apply are equal
    to the number of states.
  """
  #print("km",system.km)
  #print("kp",system.kp)
  if sum([nspin for nspin in nspins if nspin > 0]) == 0:
    raise ValueError('No electrons present!')

  #if not envelope:
  #  envelope = envelopes.make_isotropic_envelope()

  #if not feature_layer:
  #  natoms = charges.shape[0]
  #  feature_layer = make_ferminet_features(
  #      natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
  #  )

  #if isinstance(jastrow, str):
  #  if jastrow.upper() == 'DEFAULT':
  #    jastrow = jastrows.JastrowType.NONE
  #  else:
  #    jastrow = jastrows.JastrowType[jastrow.upper()]


  options = FermiNetOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      rescale_inputs=rescale_inputs,
      #envelope=envelope,
      feature_layer=feature_layer,
      #jastrow=jastrow,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=full_det,
      hidden_dims=hidden_dims,
      separate_spin_channels=separate_spin_channels,
      schnet_electron_electron_convolutions=schnet_electron_electron_convolutions,
      electron_nuclear_aux_dims=electron_nuclear_aux_dims,
      nuclear_embedding_dim=nuclear_embedding_dim,
      schnet_electron_nuclear_convolutions=schnet_electron_nuclear_convolutions,
      use_last_layer=use_last_layer,
  )

  #if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
  #  if options.bias_orbitals:
  #    raise ValueError('Cannot bias orbitals w/STO envelope.')

  #equivariant_layers = make_fermi_net_layers(nspins, charges.shape[0], options) # options contain pbc feature_layer

  #orbitals_init, orbitals_apply = make_orbitals(
  orbitals_init, orbitals_apply = make_graph_orbitals(
      nspins=nspins,
      charges=charges,
      system=system,
      options=options, # options contain pbc feature_layer
      #equivariant_layers=equivariant_layers,
  )

  def init(key: chex.PRNGKey) -> ParamTree:
    key, subkey = jax.random.split(key, num=2)
    return orbitals_init(subkey)

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network for a single datum.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute of the network evaluated at x.
    """
    #print("pos",pos.shape) # (sdim*n_par,)
    #print("spins",spins.shape) # (n_par,)

    #print("determinants",determinants)
    orbitals = orbitals_apply(params, pos, spins, atoms, charges) # list of 2 for spin up and down, each list is of shape (ndet,nup,nup), (ndet,ndown,ndown)
    #for i in range(len(orbitals)): print("orbitals i",i,orbitals[i].shape)
    #if options.states:
    #  batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
    #  orbitals = [
    #      jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
    #      for orbital in orbitals
    #  ]
    #  result = batch_logdet_matmul(orbitals)
    #else:
    #  result = network_blocks.logdet_matmul(orbitals) # this is for state=0
    result = network_blocks.logdet_matmul(orbitals) # this is for state=0; it returns phase_out, log_out; it can have w as params

    #if 'state_scale' in params:
    #  # only used at inference time for excited states
    #  result = result[0], result[1] + params['state_scale']
    return result

  return Network(
      options=options, init=init, apply=apply, orbitals=orbitals_apply
  )

def make_laughlin_net(
        nspins: tuple[int, int],
        charges: jnp.ndarray,
        system: any,
        *,
        ndim: int = 2,
        determinants: int = 1,
        states: int = 0,
        feature_layer=None,
        complex_output: bool = True,
        bias_orbitals: bool = False,
        full_det: bool = True,
        rescale_inputs: bool = False,
        # Additional parameters can be added here
) -> Network:

    if ndim != 2:
        raise ValueError("Laughlin wavefunction requires ndim=2 (2D system)")

    if not complex_output:
        raise ValueError("Laughlin wavefunction requires complex_output=True")

    # Create base options
    options = BaseNetworkOptions(
        ndim=ndim,
        system=system,
        determinants=determinants,
        states=states,
        rescale_inputs=rescale_inputs,
        feature_layer=feature_layer,
        complex_output=complex_output,
        bias_orbitals=bias_orbitals,
        full_det=full_det,
        # Add envelope and jastrow if needed
        envelope=envelopes.make_isotropic_envelope(),
        jastrow=jastrows.JastrowType.NONE
    )

    # Get Laughlin orbitals handlers
    orbitals_init, orbitals_apply = make_laughlin_orbitals(
        nspins=nspins,
        charges=charges,
        system=system,
        options=options,
    )

    def init(key: chex.PRNGKey) -> ParamTree:
        """Initializes parameters for the Laughlin wavefunction."""
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(subkey)

    def apply(
            params,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluates the Laughlin wavefunction.

        Args:
            params: Network parameters.
            pos: Electron positions.
            spins: Electron spins.
            atoms: Atom positions.
            charges: Atomic charges.

        Returns:
            Tuple of (phase, log_magnitude) of the Laughlin wavefunction.
        """
        # Get orbital matrix (contains the Laughlin state)
        orbitals = orbitals_apply(params, pos, spins, atoms, charges)

        # Convert to phase and log magnitude through determinant calculation
        result = network_blocks.logdet_matmul(orbitals)

        return result

    def apply_sym(
            params,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Symmetry-aware evaluation of the Laughlin wavefunction.

        Handles symmetry operations for quantum Hall states.

        Args:
            params: Network parameters.
            pos: Electron positions.
            spins: Electron spins.
            atoms: Atom positions.
            charges: Atomic charges.

        Returns:
            Tuple of (phase, log_magnitude) of the symmetrized Laughlin wavefunction.
        """
        # If system has specific symmetry operations defined, apply them
        if hasattr(system, 'rs') and system.rs > 1:
            # For periodic boundary conditions in quantum Hall systems
            # This assumes system.lattice and system.sym_vec are defined

            logdet_list = []
            phase_list = []

            # Loop through all symmetry-related positions
            for i in range(system.rs):
                for j in range(system.rs):
                    shift = i * system.lattice[0] + j * system.lattice[1]
                    pos_tmp = (pos.reshape(-1, 2) + shift).reshape(-1)

                    phase_tmp, log_tmp = apply(params, pos_tmp, spins, atoms, charges)

                    # Apply phase correction from symmetry vector if needed
                    if hasattr(system, 'sym_vec'):
                        phase_list.append(phase_tmp - system.sym_vec @ shift)
                    else:
                        phase_list.append(phase_tmp)

                    logdet_list.append(log_tmp)

            # Convert to complex amplitudes using the log-sum-exp trick
            phase_arr = jnp.array(phase_list)
            logdet_arr = jnp.array(logdet_list)

            maxlogdet = jnp.max(logdet_arr, axis=0)
            complex_amps = jnp.exp(1j * phase_arr + logdet_arr - maxlogdet)

            # Sum the amplitudes and convert back to phase, log format
            result = jnp.sum(complex_amps, axis=0)
            phase_out = jnp.angle(result)
            log_out = jnp.log(jnp.abs(result)) + maxlogdet

            return phase_out, log_out

        # If no symmetry operations defined, fall back to standard calculation
        return apply(params, pos, spins, atoms, charges)

    # apply_osym is the same as apply_sym for Laughlin
    apply_osym = apply_sym

    return Network(
        options=options,
        init=init,
        apply=apply,
        apply_sym=apply_sym,
        apply_osym=apply_osym,
        orbitals=orbitals_apply
    )
    
def make_bcs_net(
        nspins: Tuple[int, int],
        #Number of spin-up and spin-down electrons. 
        # This is still used even though pairing removes the need for full spin separation.
        charges: jnp.ndarray,
        #Atomic charges ¡ª usually unused in idealized BCS but kept for consistency.
        system: Any,# System config holding momentum grid, backflow, etc.
        *,
        ndim: int = 2,
        determinants: int = 1,  # BCS typically uses a single determinant
        states: int = 0,# System config holding momentum grid, backflow, etc.
        complex_output: bool = True,  # BCS wavefunctions are generally complex
        bias_orbitals: bool = False,# Whether to use bias in final orbital layer
        full_det: bool = True,# Whether to use a full determinant over all electrons
        rescale_inputs: bool = False,# Whether to apply log-scaling to distances
        feature_layer: Optional[FeatureLayer] = None,# Optional custom feature extractor
        # Additional BCS-specific parameters
        n_grid: int = 10,
) -> Network:

    if sum([nspin for nspin in nspins if nspin > 0]) == 0:
        raise ValueError('No electrons present!')

    if nspins[0] != nspins[1]:
        raise ValueError(f"BCS requires equal numbers of up and down spins, but got {nspins}")

    options = BaseNetworkOptions(
        ndim=ndim,
        determinants=determinants,
        states=states,
        complex_output=complex_output,
        bias_orbitals=bias_orbitals,
        full_det=full_det,
        rescale_inputs=rescale_inputs,
        feature_layer=feature_layer,
        system=system,
    )

    orbitals_init, orbitals_apply = make_bcs_orbitals(
        nspins=nspins,
        charges=charges,
        system=system,
        options=options,
    )

    def init(key: chex.PRNGKey) -> ParamTree:
    
        key, subkey = jax.random.split(key, num=2)# Split RNG key for parameter initialization
        return orbitals_init(subkey)# Use orbital init function to generate parameter tree

    def apply(
            params: ParamTree,# Network parameters, including delta and mu
            pos: jnp.ndarray,# Electron positions (flattened array)
            spins: jnp.ndarray,# Spin values for each electron
            atoms: jnp.ndarray,# Atom positions (unused in BCS but required by interface)
            charges: jnp.ndarray,# Nuclear charges (unused in BCS)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:# Return complex phase and log magnitude
    
        orbitals = orbitals_apply(params, pos, spins, atoms, charges)

        result = network_blocks.logdet_matmul(orbitals)

        return result

    def apply_sym(
            params: ParamTree,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray, 
            charges: jnp.ndarray, 
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    
        if hasattr(system, 'symmetry') and system.symmetry:
        
            return apply(params, pos, spins, atoms, charges)
        else:
            return apply(params, pos, spins, atoms, charges)

    return Network(
        options=options,
        init=init,
        apply=apply,
        apply_sym=apply_sym,
        apply_osym=apply_sym,
        orbitals=orbitals_apply
    )