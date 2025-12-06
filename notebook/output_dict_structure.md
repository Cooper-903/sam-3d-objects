# output 字典结构说明

本文档说明 `inference(image, mask, seed=42)` 返回的 `output` 字典的完整结构。

## 查看方式

### 方式 1: 在 Notebook 中运行代码查看

在运行 `output = inference(image, mask, seed=42)` 后，运行以下代码：

```python
# 查看所有键
print("output 字典包含的键:")
for key in sorted(output.keys()):
    value = output[key]
    if isinstance(value, torch.Tensor):
        print(f"  {key}: Tensor {list(value.shape)}")
    elif isinstance(value, list):
        print(f"  {key}: List[{len(value)}] - {type(value[0]).__name__ if value else 'empty'}")
    else:
        print(f"  {key}: {type(value).__name__}")
```

### 方式 2: 查看源代码

`output` 字典的结构在以下文件中定义：

1. **主要返回位置**: `sam3d_objects/pipeline/inference_pipeline.py`
   - `run()` 方法 (第 467-532 行): 返回合并后的字典
   - `postprocess_slat_output()` 方法 (第 534-567 行): 添加 `"gs"`, `"glb"` 等键

2. **Stage 1 输出**: `sample_sparse_structure()` 方法 (第 642-716 行)
   - 返回 `ss_return_dict`，包含稀疏结构信息

3. **位姿信息**: `sam3d_objects/pipeline/inference_utils.py`
   - `pose_decoder()` 函数 (第 224-327 行): 添加 `"rotation"`, `"translation"`, `"scale"`

4. **解码输出**: `decode_slat()` 方法 (第 589-615 行)
   - 返回 `outputs`，包含 `"gaussian"`, `"mesh"` 等

## 字典键的完整列表

### 来自 `ss_return_dict` (Stage 1: 稀疏结构采样)

| 键名 | 类型 | 说明 | 来源位置 |
|------|------|------|----------|
| `"coords"` | `torch.Tensor` | 稀疏结构的坐标，形状 `(N, 4)`，其中 `[:, 0]` 是批次索引，`[:, 1:]` 是 3D 坐标 | `sample_sparse_structure()` 第 715 行 |
| `"coords_original"` | `torch.Tensor` | 下采样前的原始坐标 | `sample_sparse_structure()` 第 704 行 |
| `"downsample_factor"` | `float` | 下采样因子 | `sample_sparse_structure()` 第 716 行 |
| `"shape"` | `torch.Tensor` | 形状潜在表示 (shape latent) | `sample_sparse_structure()` 第 695 行 |

### 来自 `pose_decoder()` (位姿解码)

| 键名 | 类型 | 说明 | 来源位置 |
|------|------|------|----------|
| `"rotation"` | `torch.Tensor` | 对象旋转（四元数），形状 `(1, 4)` | `inference_utils.py` 第 323 行 |
| `"translation"` | `torch.Tensor` | 对象平移，形状 `(1, 3)` | `inference_utils.py` 第 322 行 |
| `"scale"` | `torch.Tensor` | 对象缩放，形状 `(1, 3)` | `inference_utils.py` 第 324 行 |

### 来自 `decode_slat()` (Stage 2: 解码)

| 键名 | 类型 | 说明 | 来源位置 |
|------|------|------|----------|
| `"gaussian"` | `List[Gaussian]` | Gaussian Splat 对象列表 | `decode_slat()` 第 610 行 |
| `"mesh"` | `List[Mesh]` | 网格对象列表（如果 `decode_formats` 包含 `"mesh"`） | `decode_slat()` 第 608 行 |
| `"gaussian_4"` | `List[Gaussian]` | 4通道 Gaussian 对象列表（如果使用 4 通道解码器） | `decode_slat()` 第 612 行 |

### 来自 `postprocess_slat_output()` (后处理)

| 键名 | 类型 | 说明 | 来源位置 |
|------|------|------|----------|
| `"gs"` | `Gaussian` | `outputs["gaussian"][0]` 的别名，方便访问 | `postprocess_slat_output()` 第 562 行 |
| `"gs_4"` | `Gaussian` | `outputs["gaussian_4"][0]` 的别名（如果存在） | `postprocess_slat_output()` 第 565 行 |
| `"glb"` | `GLB` 或 `None` | GLB 格式的 3D 模型文件（如果生成了 mesh） | `postprocess_slat_output()` 第 559 行 |

### 来自 `InferencePipelinePointMap.run()` (点云相关，仅 PointMap 版本)

| 键名 | 类型 | 说明 | 来源位置 |
|------|------|------|----------|
| `"pointmap"` | `torch.Tensor` | 点云图，形状 `(H, W, 3)` | `inference_pipeline_pointmap.py` 第 376 行 |
| `"pointmap_colors"` | `torch.Tensor` | 点云颜色，形状 `(H, W, 3)` | `inference_pipeline_pointmap.py` 第 377 行 |

## 最终返回结构

最终返回的字典是 `ss_return_dict` 和 `outputs` 的合并：

```python
return {
    **ss_return_dict,  # Stage 1 的结果
    **outputs,         # Stage 2 的结果（包含后处理添加的键）
}
```

## 常用访问示例

```python
# 获取 Gaussian Splat 对象（推荐方式）
gs = output["gs"]

# 或者从列表中获取
gs = output["gaussian"][0]

# 获取位姿信息
rotation = output["rotation"]      # 四元数 (1, 4)
translation = output["translation"] # 平移 (1, 3)
scale = output["scale"]            # 缩放 (1, 3)

# 获取网格（如果生成）
if "mesh" in output:
    mesh = output["mesh"][0]

# 获取 GLB 文件（如果生成）
if "glb" in output and output["glb"] is not None:
    glb = output["glb"]
    # glb.export("output.glb")  # 导出 GLB 文件

# 获取稀疏结构坐标
coords = output["coords"]  # (N, 4) 张量
```

## 相关代码位置

- **主返回逻辑**: `sam3d_objects/pipeline/inference_pipeline.py:529-532`
- **后处理**: `sam3d_objects/pipeline/inference_pipeline.py:534-567`
- **解码**: `sam3d_objects/pipeline/inference_pipeline.py:589-615`
- **位姿解码**: `sam3d_objects/pipeline/inference_utils.py:224-327`
- **稀疏结构采样**: `sam3d_objects/pipeline/inference_pipeline.py:642-716`



