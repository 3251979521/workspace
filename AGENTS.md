# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
Temporal Knowledge Graph project for video analysis using SigLIP2 vision encoder and Neo4j.

## Key Non-Obvious Patterns

### Critical Import Warning
- **vision_encoder.py 末尾包含视频处理代码**：导入该模块时会自动执行视频读取和处理，生成 surprise_curve.png。若仅需使用函数，注释掉第27-117行代码。
- **main.py 仅导入模块**：不执行视频处理，实际处理逻辑在 vision_encoder.py 模块级执行。

### Neo4j Connection
- `tkg_manager.py` 第52行硬编码了 Neo4j 凭据：`bolt://localhost:7687`, `neo4j`, `zy159632`。需要修改为环境变量或配置。
- 使用异步驱动 `AsyncGraphDatabase.driver()` 和 `session.run()` 模式。

### Surprise Detection
- 默认阈值 `threshold = 0.15`（vision_encoder.py 第41行）
- surprise 值高于阈值表示场景切换或显著变化（标记为 "CHUNK BOUNDARY"）

## Architecture
- `model/vision_encoder.py`: SigLIP2 特征提取 + surprise 计算
- `tkg_manager.py`: TemporalKGManager 异步图谱写入
- `main.py`: 入口点（仅导入）