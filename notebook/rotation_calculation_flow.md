# rotation 计算流程说明

## 问题

在 `inference_pipeline_pointmap.py` 的 `run` 方法（第 316-423 行）中，没有直接看到 `rotation` 的计算代码。这是因为 `rotation` 是通过 `pose_decoder` 函数从模型输出中提取并转换得到的。

## 完整流程

### 1. Stage 1: 稀疏结构采样 (`sample_sparse_structure`)

**位置**: `sam3d_objects/pipeline/inference_pipeline.py:642-719`

```python
# 第 686-691 行：调用 ss_generator 生成稀疏结构
return_dict = ss_generator(
    latent_shape_dict,
    image.device,
    *condition_args,
    **condition_kwargs,
)
```

**`ss_generator` 返回的 `return_dict` 包含**:
- `"shape"`: 形状潜在表示
- `"quaternion"` 或 `"6drotation"` 或 `"6drotation_normalized"`: 旋转信息（由模型生成）
- `"translation"`: 平移信息（由模型生成）
- `"scale"`: 缩放信息（由模型生成）
- 其他键...

**注意**: 这些键的具体名称取决于模型配置和 `pose_target_convention`。

### 2. 位姿解码 (`pose_decoder`)

**位置**: `sam3d_objects/pipeline/inference_pipeline_pointmap.py:360-365`

```python
ss_return_dict.update(
    self.pose_decoder(
        ss_return_dict,  # 包含 ss_generator 的输出
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )
)
```

**`pose_decoder` 的实现**: `sam3d_objects/pipeline/inference_utils.py:224-327`

#### 步骤 2.1: 键映射 (第 231-244 行)

```python
key_mapping = {
    "shape": "x_shape_latent",
    "quaternion": "x_instance_rotation",
    "6drotation": "x_instance_rotation_6d",
    "6drotation_normalized": "x_instance_rotation_6d_normalized",
    "translation": "x_instance_translation",
    "scale": "x_instance_scale",
    "translation_scale": "x_translation_scale",
}

# 将 ss_return_dict 中的键映射到标准名称
pose_target_dict = {}
for k, v in x.items():
    pose_target_dict[key_mapping.get(k, k)] = v
```

#### 步骤 2.2: 旋转格式转换 (第 247-281 行)

如果模型输出的是 6D rotation，需要转换为 quaternion：

```python
# 如果存在 6D rotation，转换为旋转矩阵，再转换为 quaternion
if "x_instance_rotation_6d" in pose_target_dict or "x_instance_rotation_6d_normalized" in pose_target_dict:
    # 提取两个 3D 向量
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    
    # 归一化第一个向量
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    
    # 使第二个向量与第一个正交
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    
    # 计算第三个向量（叉积）
    b3 = torch.cross(b1, b2, dim=-1)
    
    # 堆叠成旋转矩阵
    rotation_matrix = torch.stack([b1, b2, b3], dim=-1)
    
    # 转换为 quaternion
    quaternion = matrix_to_quaternion(rotation_matrix)
    pose_target_dict["x_instance_rotation"] = quaternion
```

#### 步骤 2.3: 缩放处理 (第 283-291 行)

```python
# 对 scale 应用 exp 函数（如果模型输出的是 log scale）
if "x_instance_scale" in pose_target_dict:
    pose_target_dict["x_instance_scale"] = torch.exp(
        pose_target_dict["x_instance_scale"]
    )
```

#### 步骤 2.4: 转换为实例位姿 (第 312-320 行)

```python
# 使用 PoseTargetConverter 将目标位姿转换为实例位姿
pose_instance_dict = PoseTargetConverter.dicts_pose_target_to_instance_pose(
    pose_target_convention=pose_target_convention,
    x_instance_scale=pose_target_dict["x_instance_scale"],
    x_instance_translation=pose_target_dict["x_instance_translation"],
    x_instance_rotation=pose_target_dict["x_instance_rotation"],
    x_translation_scale=pose_target_dict["x_translation_scale"],
    x_scene_scale=pose_target_dict["x_scene_scale"],
    x_scene_center=pose_target_dict["x_scene_center"],
)
```

#### 步骤 2.5: 返回最终结果 (第 321-325 行)

```python
return {
    "translation": pose_instance_dict["instance_position_l2c"].squeeze(0),
    "rotation": pose_instance_dict["instance_quaternion_l2c"].squeeze(0),  # ← 这里！
    "scale": pose_instance_dict["instance_scale_l2c"].squeeze(0).mean(-1, keepdim=True).expand(1,3),
}
```

### 3. 更新到 `ss_return_dict`

**位置**: `sam3d_objects/pipeline/inference_pipeline_pointmap.py:360-365`

```python
ss_return_dict.update(
    self.pose_decoder(ss_return_dict, ...)
)
# 现在 ss_return_dict 包含了 "rotation", "translation", "scale"
```

### 4. 最终返回

**位置**: `sam3d_objects/pipeline/inference_pipeline_pointmap.py:418-423`

```python
return {
    **ss_return_dict,  # 包含 rotation, translation, scale
    **outputs,         # 包含 gaussian, mesh, gs, glb 等
    "pointmap": ...,
    "pointmap_colors": ...,
}
```

## 关键代码位置总结

| 步骤 | 位置 | 说明 |
|------|------|------|
| 1. 模型生成位姿数据 | `inference_pipeline.py:686-691` | `ss_generator()` 返回包含 `quaternion`/`6drotation` 等的字典 |
| 2. 调用 pose_decoder | `inference_pipeline_pointmap.py:360-365` | `self.pose_decoder(ss_return_dict)` |
| 3. 键映射 | `inference_utils.py:231-244` | 将模型输出键映射到标准名称 |
| 4. 旋转转换 | `inference_utils.py:247-281` | 6D rotation → 旋转矩阵 → quaternion |
| 5. 位姿转换 | `inference_utils.py:312-320` | 使用 `PoseTargetConverter` 转换 |
| 6. 返回 rotation | `inference_utils.py:323` | 返回 `"rotation"` 键 |
| 7. 更新字典 | `inference_pipeline_pointmap.py:360` | `ss_return_dict.update(...)` |

## 为什么在 `run` 方法中看不到 rotation 的计算？

因为 `rotation` 的计算逻辑被封装在 `pose_decoder` 函数中，而 `pose_decoder` 是在 `inference_utils.py` 中定义的。`run` 方法只是调用它：

```python
# 第 360-365 行：调用 pose_decoder，它会从 ss_return_dict 中提取并转换位姿信息
ss_return_dict.update(
    self.pose_decoder(
        ss_return_dict,  # 输入：包含模型原始输出的字典
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )  # 输出：包含 "rotation", "translation", "scale" 的字典
)
```

## 查看 rotation 的实际值

在 notebook 中运行：

```python
output = inference(image, mask, seed=42)
print("rotation:", output["rotation"])
print("translation:", output["translation"])
print("scale:", output["scale"])
```



