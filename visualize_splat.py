#!/usr/bin/env python3
"""
使用 PyTorch3D 可视化 splat.ply 点云文件
"""

import torch
import matplotlib.pyplot as plt
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import numpy as np

def load_and_visualize_ply(ply_path, device='cuda', interactive=True):
    """
    加载 PLY 文件并使用 PyTorch3D 可视化
    
    Args:
        ply_path: PLY 文件路径
        device: 设备 ('cuda' 或 'cpu')
        interactive: 是否使用交互式 Plotly 可视化
    """
    print(f"正在加载 {ply_path}...")
    
    # 使用 PyTorch3D 的 IO 加载点云
    io = IO()
    try:
        # 尝试作为点云加载
        pointcloud = io.load_pointcloud(ply_path, device=device)
        print(f"✓ 成功加载点云: {len(pointcloud.points_packed())} 个点")
    except Exception as e:
        print(f"作为点云加载失败: {e}")
        print("尝试作为 mesh 加载...")
        # 如果失败，尝试作为 mesh 加载
        mesh = io.load_mesh(ply_path, device=device)
        # 从 mesh 提取点云
        verts = mesh.verts_packed()
        # 如果有颜色信息，使用它；否则使用默认颜色
        if hasattr(mesh, 'textures') and mesh.textures is not None:
            # 尝试获取顶点颜色
            try:
                colors = mesh.textures.verts_features_packed()
                if colors.shape[1] >= 3:
                    colors = colors[:, :3]
                else:
                    colors = torch.ones_like(verts) * 0.7
            except:
                colors = torch.ones_like(verts) * 0.7
        else:
            colors = torch.ones_like(verts) * 0.7
        
        pointcloud = Pointclouds(points=[verts], features=[colors])
        print(f"✓ 从 mesh 提取点云: {len(verts)} 个点")
    
    # 获取点云数据
    points = pointcloud.points_packed()
    
    # 检查是否有颜色信息
    features = pointcloud.features_packed()
    if features is not None:
        if features.shape[1] >= 3:
            colors = features[:, :3]
            # 确保颜色值在 [0, 1] 范围内
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = torch.ones_like(points) * 0.7
    else:
        # 使用默认颜色（根据 z 坐标着色）
        z_coords = points[:, 2]
        z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
        colors = plt.cm.viridis(z_normalized.cpu().numpy())[:, :3]
        colors = torch.from_numpy(colors).float().to(device)
    
    # 更新点云的颜色
    pointcloud = Pointclouds(points=[points], features=[colors])
    
    print(f"点云范围:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    if interactive:
        # 使用 Plotly 交互式可视化
        print("\n正在生成交互式可视化...")
        fig = plot_scene({
            "点云": {
                "splat": pointcloud
            }
        })
        fig.show()
        print("✓ 交互式可视化已打开")
    else:
        # 使用渲染器生成静态图像
        print("\n正在渲染静态图像...")
        render_pointcloud_image(pointcloud, device=device)
    
    return pointcloud

def render_pointcloud_image(pointcloud, device='cuda', image_size=800, output_path='splat_rendered.png'):
    """
    渲染点云为静态图像
    
    Args:
        pointcloud: Pointclouds 对象
        device: 设备
        image_size: 图像尺寸
        output_path: 输出路径
    """
    # 设置相机
    # 计算点云的中心和范围
    points = pointcloud.points_packed()
    center = points.mean(dim=0)
    scale = (points.max(dim=0)[0] - points.min(dim=0)[0]).max()
    
    # 设置相机位置（从不同角度观察）
    distance = scale * 2.5
    elevation = 10.0
    azimuth = 45.0
    
    # 转换为弧度
    elevation_rad = np.radians(elevation)
    azimuth_rad = np.radians(azimuth)
    
    # 计算相机位置
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
    
    # 设置渲染器
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.003,
        points_per_pixel=10,
    )
    
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
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Splat Point Cloud Visualization')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 渲染图像已保存到: {output_path}")
    plt.close()

if __name__ == "__main__":
    import sys
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # PLY 文件路径
    ply_path = "splat.ply"
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
    
    # 加载和可视化
    pointcloud = load_and_visualize_ply(
        ply_path, 
        device=device, 
        interactive=True  # 设置为 False 可以生成静态图像
    )
    
    print("\n可视化完成！")

