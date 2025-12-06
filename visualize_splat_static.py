#!/usr/bin/env python3
"""
使用 PyTorch3D 渲染 splat.ply 点云为静态图像
"""

import torch
import matplotlib.pyplot as plt
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import numpy as np

def render_pointcloud_views(ply_path, output_dir="rendered_views", device='cuda', image_size=1024):
    """
    从多个角度渲染点云并保存图像
    
    Args:
        ply_path: PLY 文件路径
        output_dir: 输出目录
        image_size: 图像尺寸
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在加载 {ply_path}...")
    io = IO()
    pointcloud = io.load_pointcloud(ply_path, device=device)
    print(f"✓ 成功加载点云: {len(pointcloud.points_packed())} 个点")
    
    # 获取点云数据
    points = pointcloud.points_packed()
    features = pointcloud.features_packed()
    
    # 处理颜色
    if features is not None and features.shape[1] >= 3:
        colors = features[:, :3]
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        # 使用 z 坐标着色
        z_coords = points[:, 2]
        z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
        colors = plt.cm.viridis(z_normalized.cpu().numpy())[:, :3]
        colors = torch.from_numpy(colors).float().to(device)
    
    pointcloud = Pointclouds(points=[points], features=[colors])
    
    # 计算点云的中心和范围
    center = points.mean(dim=0)
    scale = (points.max(dim=0)[0] - points.min(dim=0)[0]).max()
    distance = scale * 2.5
    
    # 设置多个视角
    views = [
        {"elevation": 10, "azimuth": 0, "name": "front"},
        {"elevation": 10, "azimuth": 90, "name": "right"},
        {"elevation": 10, "azimuth": 180, "name": "back"},
        {"elevation": 10, "azimuth": 270, "name": "left"},
        {"elevation": 45, "azimuth": 45, "name": "top_right"},
        {"elevation": -20, "azimuth": 45, "name": "bottom_right"},
    ]
    
    # 设置渲染器
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.003,
        points_per_pixel=10,
    )
    
    for view in views:
        print(f"正在渲染视角: {view['name']}...")
        
        # 计算相机位置
        elevation_rad = np.radians(view["elevation"])
        azimuth_rad = np.radians(view["azimuth"])
        
        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        
        camera_position = torch.tensor([[x, y, z]], device=device) + center.unsqueeze(0)
        
        # 创建相机
        cameras = PerspectiveCameras(
            device=device,
            R=torch.eye(3, device=device).unsqueeze(0),
            T=-camera_position,
            focal_length=torch.tensor([[image_size * 1.2]], device=device),
            image_size=torch.tensor([[image_size, image_size]], device=device),
        )
        
        # 创建渲染器
        rasterizer = PointsRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )
        
        # 渲染
        images = renderer(pointcloud)
        
        # 转换为 numpy 并保存
        image = images[0, ..., :3].cpu().numpy()
        image = np.clip(image, 0, 1)
        
        output_path = os.path.join(output_dir, f"splat_{view['name']}.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Splat Point Cloud - {view["name"]} View')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已保存到: {output_path}")
    
    print(f"\n✓ 所有视角已渲染完成，保存在 {output_dir}/ 目录")

if __name__ == "__main__":
    import sys
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    ply_path = "splat.ply"
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
    
    render_pointcloud_views(ply_path, device=device)








