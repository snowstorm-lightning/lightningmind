# LightningMind - 安装指南

本项目使用 [uv](https://docs.astral.sh/uv/) 作为包管理器，以确保依赖项的高效管理和环境的一致性。

## 环境要求

* **Python 版本**：需要 Python **3.14** 或更高版本。
* **uv**：建议先安装 `uv`。如果尚未安装，请参考 [uv 官方安装指南](https://docs.astral.sh/uv/getting-started/installation/)。

## 安装步骤


### 1. 克隆项目
首先，将项目克隆到本地：
```bash
git clone [https://github.com/snowstorm-lightning/lightningmind.git](https://github.com/snowstorm-lightning/lightningmind.git)
cd lightningmind
```

### 2. 同步环境
在项目根目录下运行以下命令。`uv` 会根据 `.python-version` 和 `pyproject.toml` 自动创建虚拟环境并安装所有依赖项：

```bash
uv sync
```

### 3.将相关数据集和权重都下载到本地的dataset和checkpoints文件夹中

### 4.运行代码
```bash
uv run eval_llm.py
```
