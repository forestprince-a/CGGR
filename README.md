# CGGR — Consensus-Guided Graph Refinement



- `model.py`：主模型与交替优化。
- `graph.py`、`projection.py`、`symmetric_projection.py`：构图、图子问题与约束投影。
- `fusion.py`：模态谱表示、旋转、共识和权重更新。
- `data.py`：配对 QC 及 RNA / ADT / ATAC 预处理。
- `gpu_graph.py`、`gpu_spectrum.py`、`backend.py`：GPU 求解与计算后端。
- `config.py`、`model_config.json`：模型参数。

模型目标为 `J = L_graph + alpha * L_bridge + beta * L_fusion`。三个图损失系数固定为 1；`alpha`、`beta` 必须为正。固定使用 300 轮、30 个候选邻居、每个图每轮 5 步 ADMM、模态权重下界 `0.1/V` 和 20 次 K-means++ 初始化。其余数值常量见 `config.py`，与附录的基础协议一致。

图初始化、S/C 更新、Q 与秩罚系数更新、H 更新、R/F 更新和模态权重更新按 Algorithm S1 执行。预处理按 S1.1：RNA 特征至少出现在 3 个细胞，ADT、ATAC 和 gene activity 至少出现在 1 个细胞；采用共同细胞掩码。主模型不接收参考标签。

安装：`python -m pip install -r requirements.txt`；CUDA 12 环境使用 `requirements-gpu.txt`。

从本目录调用，输入各模态已预处理且细胞顺序一致的 N×D 矩阵：

```python
import json
import numpy as np
from cggr.config import ModelConfig
from cggr.model import CGGR

with open("model_config.json", encoding="utf-8") as f:
    params = json.load(f)
params.update(clusters=6, device="cpu", random_state=0)
views = [np.load("rna_embedding.npy"), np.load("adt_embedding.npy")]
model = CGGR(ModelConfig(**params)).fit(views)
labels = model.labels_
embedding = model.cluster_embedding_
```

`clusters` 按任务指定，需满足 `2 <= K <= N//2`，且至少有 32 个细胞。输入为经过 S1.1 预处理、行范数为 1 的配对表示。`labels_` 为从 1 开始的聚类编号；`consensus_` 保留列正交共识，`cluster_embedding_` 为单独的行归一化表示。GPU 使用 `device="cuda:0"`，基础配置默认 CUDA、float64。

`history_` 保存模型损失和数值诊断；`states_` 保存各模态图和谱变量；`weights_` 保存模态权重。训练结束后直接对共识表示行归一化并执行 K-means
