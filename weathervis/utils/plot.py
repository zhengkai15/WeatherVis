import os
from sys import prefix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

from functools import wraps
import time

def time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper




def add_provence(ax,geo_path):
    with open(geo_path) as src:
        context = ''.join([l for l in src if not l.startswith('#')])
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
    for line in borders:
        ax.plot(line[0::2], line[1::2], '-', lw=0.5, color='k',
                transform=ccrs.PlateCarree()) 
    return ax


# from weatherdata.model.open import transform_concat
# from weatherdata.model.open import update_dims_era5 as update_dims_
# import xarray as xr
# sea_mask = xr.open_dataset("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/fuxi-c88-eval/20241123.nc")
# sea_mask = update_dims_(sea_mask)
# sea_mask = transform_concat(sea_mask, var_name="sst").isel(time=0, variable=0)
# land_mask = xr.where(sea_mask.notnull(), 0, 1).isel(channel=0)

import regionmask
from .maskout import shp2clip
@time_decorator
def plot(ds, save_path="./",  level=None, geo_path="", var="tp", title=None):
    # ds = ds*(land_mask.drop_vars("time"))
    # mask = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(ds)
    # ds = ds.where((mask == 0))
    
    time = pd.to_datetime(ds.time.values).strftime("%Y%m%d")
    init_time = pd.to_datetime(ds.init_time.values).strftime("%Y%m%d%H")
    if title is None:
        title = f"{init_time}_{time}"
    # 创建地图投影
    proj = ccrs.PlateCarree()
    
    # 创建图形并添加地图
    plt.figure(figsize=(10, 6), dpi=100)
    ax = plt.axes(projection=proj)


    # 创建颜色映射和norm
    if var == 'tp':
        colors1 = [
            "#8B4513",  # 深棕色 (非常干旱)
            "#CD853F",  # 亮棕色
            "#DEB887",  # 米色
            "#EEE8AA",  # 浅卡其色
            "#C0D890",  # 淡绿黄
            "#90EE90",  # 浅绿色
            "#32CD32",  # 酸橙绿
            "#006400"   # 深绿色 (非常湿润)
            ]
        # levels1 = [-100, -80, -50, -20, 0, 20, 50, 100, 200]  # 9 个分界点 -> 8 个颜色区间
        levels1 = [-100, -80, -50, -20, 0, 20, 50, 100]  # 9 个分界点 -> 8 个颜色区间  panchen
        levels1 = [-80, -60, -40, -20, 0, 20, 40, 60, 80]  # 9 个分界点 -> 8 个颜色区间  panchen
        cblabel="%"
        cmap = mcolors.ListedColormap(colors1)
        norm = mcolors.BoundaryNorm(levels1, len(colors1))
    elif var == 't2m':
        # 自定义颜色：紫-红-白-蓝
        cblabel="℃"
        colors2 = [
            "#00008B", 
            "#0000FF", 
            "#87CEFA", 
            "#ADD8E6", 
            "#F5F5DC", 
            "#FFDAB9", 
            "#FA8072", 
            "#FF6347", 
            "#B22222", 
            "#8B008B"
        ]
        # levels2 = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]  # 11 个分界点 -> 10 个颜色区间 panchen
        levels2 = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]  # 11 个分界点 -> 10 个颜色区间
        cmap = mcolors.ListedColormap(colors2)
        norm = mcolors.BoundaryNorm(levels2, len(colors2))
    else:
        raise ValueError("Invalid variable type. Please choose 'tp' or 't2m'.")

    # 绘制等高线图
    contour = ds.plot.contourf(ax=ax, levels=level, cmap=cmap, norm=norm, add_colorbar=False)

    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='white')
    # ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='none')
    # 添加经纬网格线
    gridlines = ax.gridlines(draw_labels=True, crs=proj, linestyle='--')
    gridlines.top_labels = False   # 不显示顶部标签
    gridlines.right_labels = False # 不显示右侧标签
    gridlines.left_labels = True   # 显示左侧标签
    gridlines.bottom_labels = True # 显示底部标签
    gridlines.xlocator = plt.MultipleLocator(10)  # 经度线间隔 5 度
    gridlines.ylocator = plt.MultipleLocator(10)  # 纬度线间隔 5 度
    gridlines.xlabel_style = {'size': 10, 'color': 'gray'}
    gridlines.ylabel_style = {'size': 10, 'color': 'gray'}
    # 添加颜色条
    posn = ax.get_position()
    # 设置颜色条的位置和大小
    cax = plt.axes([posn.x1-0.08, posn.y0, 0.01, posn.height])  # 颜色条放在图的右边，并和图像等高
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', cax=cax, extend='both')
    cbar.set_label(cblabel, rotation=0)  # 添加颜色条标签并设置为垂直方向
    
    # TODO mask china
    # https://github.com/GaryBikini/ChinaAdminDivisonSHP.git  Country and rename to china
    shpfile_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/WeatherVis/weathervis/utils/china_shp/china.shp"
    # shapefile.Reader(shpfile)[0].record :['100000', '中华人民共和国']
    shp2clip(contour, ax, shpfile=shpfile_path, proj = proj, clabel = None, vcplot = False)

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax = add_provence(ax, geo_path)
    # ax.set_extent([80, 134, 20, 48])
    ax.set_extent([72, 136, 5, 53])
    if save_path is not None:
        save_path = os.path.join(save_path, init_time, f"{var}_{time}.png")
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=100, format='png')
        plt.close()
    else:
        return ax
    
    


import os
import imageio
from PIL import Image

def create_gif_from_images(image_paths, gif_output_path, duration=1):
    """
    将多个图片合并成 GIF 动画，并保存为文件。

    Parameters:
    - image_paths (list of str): 包含图片路径的列表
    - gif_output_path (str): 输出 GIF 文件的路径
    - duration (float): 每帧之间的持续时间，单位为秒
    """
    # 读取所有图片并转换为 RGB 格式
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')  # 确保是RGB格式
        images.append(img)
    
    # 使用 imageio 保存为 gif
    imageio.mimsave(gif_output_path, images, duration=duration)
    print(f"GIF已生成: {gif_output_path}")