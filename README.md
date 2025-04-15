# MNIST 训练实验

本项目是一个基于PyTorch的MNIST数字识别实验模板，支持在多种环境下进行训练和结果比较。

## 功能特性

- 使用PyTorch训练MNIST手写数字识别模型
- 支持CPU和CUDA环境下的训练
- 自动为不同环境的训练结果和模型创建独立目录
- 集成MLflow进行实验追踪和结果可视化
- 提供数据哈希验证确保实验可重现性

## 环境设置

本项目使用[Pixi](https://github.com/prefix-dev/pixi)进行环境管理，支持两种环境：

- **default**: CPU环境
- **cuda**: GPU加速环境

### 安装依赖

```bash
# 安装基础环境
pixi install

# 或安装CUDA环境
pixi install --environment cuda
```

## 数据准备

```bash
# 下载并准备MNIST数据集
pixi run prepare-data
```

## 训练模型

关键特性是可以在不同环境中运行训练，结果会保存到对应环境的子目录中。

### 默认(CPU)环境训练

```bash
# 使用CPU环境训练
pixi run train
```

训练结果将保存在：
- 模型: `models/default/mnist_cnn.pt`
- 结果数据: `results/default/mnist_results.json`

### CUDA(GPU)环境训练

```bash
# 使用CUDA环境训练
pixi run --environment cuda train
```

训练结果将保存在：
- 模型: `models/cuda/mnist_cnn.pt`
- 结果数据: `results/cuda/mnist_results.json`

## 结果比较

训练完成后，您可以比较不同环境下的训练结果：

```bash
# 比较两个环境的结果文件
diff results/default/mnist_results.json results/cuda/mnist_results.json
```

## 其他任务

```bash
# 清理所有数据、结果和模型
pixi run clean-all

# 仅清理结果
pixi run clean-results

# 验证数据哈希值(确保数据一致性)
pixi run verify-data-hash
```

## 多平台验证

请参考 [MULTIPLATFORM_VERIFICATION.md](MULTIPLATFORM_VERIFICATION.md) 获取关于在多平台上验证实验可重现性的信息。

## 结果分析

训练过程中的所有指标都通过MLflow记录，可以通过以下命令启动MLflow UI：

```bash
mlflow ui --backend-store-uri ./mlruns
```

然后在浏览器中访问 http://localhost:5000 查看训练结果。

## 常见问题

### 为什么CUDA和CPU环境的结果有差异？

不同执行环境下的结果可能有微小差异，这是正常的，主要原因包括：

1. 浮点数运算在GPU和CPU上的实现差异
2. CUDA库的优化可能导致计算路径不同
3. 不同环境下的随机数生成微小差异

如果需要更严格的结果一致性，可以考虑：
- 使用双精度浮点数
- 避免依赖特定环境优化的操作
- 更严格地控制随机种子 

## 文档

本项目提供了全面的文档，包括：

- **主文档**：使用说明、API参考和贡献指南
- **项目展示页面**：项目特性和结果展示
- **研究论文**：详细的方法论和实验结果

### 查看在线文档

文档已通过GitHub Pages发布，可以访问：
https://[username].github.io/mnist/

### 本地构建文档

要在本地构建和查看文档：

```bash
# 安装Quarto
# 从 https://quarto.org/docs/get-started/ 下载并安装

# 安装Quarto扩展
cd docs
quarto add quarto-ext/fontawesome
quarto add grantmcdermott/quarto-revealjs-clean
quarto add mikemahoney218/quarto-arxiv

# 渲染主文档
quarto render

# 查看生成的文档
open _output/index.html
```

更多关于文档的详细说明请参考 [docs/README.md](docs/README.md)。 