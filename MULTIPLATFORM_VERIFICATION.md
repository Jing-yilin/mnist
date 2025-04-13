# MNIST模型多平台验证指南

本文档介绍如何在多个平台上验证MNIST模型的可复现性。

## 支持的平台

当前支持以下平台进行验证：

- Windows (CPU)
- Linux (CPU)
- macOS (CPU)
- Linux (CUDA/GPU)

## 本地验证

### 前提条件

- 已安装[Pixi](https://github.com/prefix-dev/pixi)
- 各平台上具有必要的环境（对于CUDA验证需要支持CUDA的GPU和相应驱动）

### 执行验证

在代码库根目录下执行以下命令：

```bash
# 对应平台的验证
pixi run verify-windows  # 在Windows上
pixi run verify-linux    # 在Linux上
pixi run verify-macos    # 在macOS上
pixi run verify-linux-cuda  # 在支持CUDA的Linux上

# 执行报告生成
pixi run create-verification-report
```

执行后，会在项目根目录下生成`verification_report.md`文件，包含详细的验证结果。

### 查看验证日志

所有验证日志存储在`logs/`目录中，每个日志文件包含完整的平台信息和验证状态。

## 使用GitHub Actions进行验证

本项目包含GitHub Actions工作流，可自动化在多个平台上进行验证：

1. `.github/workflows/reproduction.yml` - 在多个平台上运行验证
2. `.github/workflows/status-summary.yml` - 生成验证状态汇总

### 启动验证

您可以通过以下方式启动验证：

1. 推送到`main`分支
2. 创建针对`main`分支的Pull Request
3. 手动在GitHub界面上触发工作流（通过"Actions"选项卡）

### 查看验证结果

验证完成后，可以通过以下方式查看结果：

1. 在GitHub "Actions"选项卡中查看工作流执行状态
2. 下载生成的工件（artifacts）查看详细的验证报告和日志

## 解决常见问题

### 数据哈希验证失败

如果数据哈希验证失败，可能是由于：

1. 数据文件损坏或不完整
2. 数据下载过程中发生错误

解决方法：清除数据并重新下载

```bash
pixi run clean-data
pixi run prepare-data
pixi run update-data-hash  # 更新哈希值
```

### 模型训练结果不一致

如果模型训练结果在不同平台上不一致，可能是由于：

1. 随机数种子未固定
2. 不同平台上的运算库版本不一致

解决方法：检查`train.py`中的随机数种子设置，确保所有库版本一致。

## 高级使用

### 自定义验证过程

您可以通过修改`pixi.toml`中的任务定义来自定义验证过程。例如，添加新的特定平台验证任务：

```toml
[tasks.verify-custom-platform]
cmd = "python scripts/platform_verification.py log --message 'Custom platform verification'"
depends-on = ["reproduction"]
```

### 添加新平台支持

要添加新平台支持，需要：

1. 在`pixi.toml`中更新`platforms`列表
2. 添加新的验证任务
3. 更新GitHub Actions工作流定义

## 贡献指南

欢迎对多平台验证流程提出改进建议或贡献代码。请遵循以下步骤：

1. Fork本仓库
2. 创建功能分支
3. 提交更改
4. 创建Pull Request 