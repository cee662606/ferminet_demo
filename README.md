# FermiNet-Demo: 深度神经网络模拟多电子量子系统

本项目为 FermiNet 模型的简化版本展示，采用变分蒙特卡洛（VMC）方法，通过深度神经网络近似多体电子系统的基态波函数。适用于如氢分子、锂原子等简单体系的模拟演示与教学展示。

> 原始论文：Pfau et al., *Phys. Rev. Research* 2, 033429 (2020)

## 🧠 项目亮点

- 使用反对称神经网络结构逼近多体费米子波函数
- 支持通过配置文件定义不同原子/分子的结构与电子数
- 可扩展至激发态计算（如 NES-VMC 方法）
- 支持 GPU 并行加速

## 🔧 快速运行

推荐使用 `conda` 或 `virtualenv` 建立独立环境并安装：

```bash
pip install -e .
```

运行一个简化示例（如 Li 原子）：

```bash
python run.py --config configs/atom.py --system.atom Li
```

或自行构建如氢分子 H₂ 的配置：

```python
cfg.system.electrons = (1, 1)
cfg.system.molecule = [Atom('H', (0, 0, -1)), Atom('H', (0, 0, 1))]
```

更多细节可参考 `base_config.py` 与 `train.py` 文件。

## 📁 项目结构

| 文件/目录         | 说明                           |
|------------------|------------------------------|
| `ferminet/`      | 网络结构与核心计算逻辑             |
| `configs/`       | 示例配置文件（Li、H₂等）         |
| `train.py`       | 训练入口，核心逻辑               |
| `plot.py`        | 波函数或轨道可视化支持              |
| `README.md`      | 当前说明文档                     |

## 📊 输出说明

训练完成后将生成：
- `train_stats.csv`：每轮迭代的能量与接受率
- `checkpoints/`：模型参数存档
- 可选 `.npy` 文件用于轨道与波函数可视化分析

## 📎 说明

本仓库为教学与展示目的，代码经过简化，非原始 FermiNet 全功能实现。若需完整功能或扩展至大体系，请参阅 DeepMind 官方版本或相关论文。
