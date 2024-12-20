import os
from sys import prefix
from matplotlib.patches import PathPatch
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


def add_provence(ax, geo_path, transform=ccrs.PlateCarree()):
    """
    Adds province borders to the given axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to which the province borders will be added.
    geo_path (str): The path to the shapefile containing the province borders.
    geo_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/data/DEM/china-geospatial-data-UTF8/CN-border-La.gmt"

    Returns:
    matplotlib.axes.Axes: The axes with the province borders added.
    """
    with open(geo_path) as src:
        context = ''.join([l for l in src if not l.startswith('#')])
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
    for line in borders:
        ax.plot(line[0::2], line[1::2], '-', lw=1, color='k',
                transform=ccrs.PlateCarree()) 
    return ax

# TODO 使用这个链接里面的shp文件重新写一个添加省界的方法
# https://github.com/GaryBikini/ChinaAdminDivisonSHP.git 
# 定义绘制省界的函数
import shapefile
def plot_shapefile(ax, shapefile_path):
    '''
    shapefile_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/WeatherVis/ChinaAdminDivisonSHP/2. Province/province.shp"
    '''
    
    '''
    # 有不规则直线
    # 读取shapefile文件
    sf = shapefile.Reader(shapefile_path)
    
    for shape in sf.shapes():
        # 获取边界点
        points = shape.points
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, 'k', linewidth=1)
    '''
    
    '''
    # 有不规则直线
    # 读取shapefile文件
    sf = shapefile.Reader(shapefile_path)
    # 读取 Shapefile 中的线条
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    for shape in sf.shapes():
        # 获取每个多边形的点坐标
        points = shape.points
        # 如果数据是封闭的，可以通过忽略最后一个点避免首尾相连
        if points[0] == points[-1]:  # 判断路径是否封闭
            points = points[:-1]  # 删除最后一个点，避免封闭
        # 路径代码
        codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)

        # 创建路径对象
        path = Path(points, codes)

        # 绘制路径
        patch = PathPatch(path, edgecolor='black', facecolor='none', lw=1)
        ax.add_patch(patch)
    '''
    '''
    # 有不规则直线
    # 遍历 shapefile 中的每个形状
    from matplotlib.patches import Polygon
    sf = shapefile.Reader(shapefile_path)
    for shape in sf.shapes():
        # 获取每个多边形的坐标
        points = shape.points
        # 如果路径是封闭的，去掉最后一个点（重复的第一个点）
        if points[0] == points[-1]:
            points = points[:-1]

        # 将路径数据转化为 Polygon 并绘制
        polygon = Polygon(points, edgecolor='black', facecolor='none', linewidth=1)
        ax.add_patch(polygon)    
    '''
    # 使用gdf读取
    import geopandas as gpd
    # Since LambertConformal gridlines are not supported, we will use a workaround.
    # Convert the shapefile to a GeoDataFrame with PlateCarree projection.
    gdf = gpd.read_file(shapefile_path)
    # gdf = gdf.to_crs(epsg=4326)  # EPSG 4326 is the PlateCarree projection
    # gdf = gdf.to_crs(epsg=3347)  # EPSG 3347 is the 兰博拖投影 会卡住; 去掉会不绘制线条
    gdf.boundary.plot(ax=ax, linewidth=1, edgecolor='black')
    return ax 
        

def get_mask_from_era5():
    """
    Extracts a land mask from the ERA5 dataset.

    This function reads the ERA5 dataset from the specified path, updates the dimensions,
    transforms the concatenated data, and then creates a land mask by setting sea areas to 0
    and land areas to 1.

    Returns:
    xarray.DataArray: A 2D DataArray representing the land mask.
    """
    from weatherdata.model.open import transform_concat
    from weatherdata.model.open import update_dims_era5 as update_dims_
    import xarray as xr

    # Open the ERA5 dataset
    sea_mask = xr.open_dataset("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/fuxi-c88-eval/20241123.nc")

    # Update the dimensions of the dataset
    sea_mask = update_dims_(sea_mask)

    # Transform the concatenated data and select the first time step and variable
    sea_mask = transform_concat(sea_mask, var_name="sst").isel(time=0, variable=0)

    # Create a land mask by setting sea areas to 0 and land areas to 1
    land_mask = xr.where(sea_mask.notnull(), 0, 1).isel(channel=0)

    return land_mask

        
        

import regionmask
from .maskout import shp2clip

def mask_land_with_regionmask(ds):
    mask = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(ds)
    ds = ds.where((mask == 0))
    return ds

def mask_land_with_era5_mask(ds):
    land_mask = get_mask_from_era5()
    ds = ds*(land_mask.drop_vars("time"))
    return ds


@time_decorator
def plot(ds, save_path="./",  levels=None, geo_path="", var="tp", title=None):
    
    # # Apply the land mask from the ERA5 dataset to the dataset.
    # ds = mask_land_with_era5_mask(ds)
    
    # # Apply the land mask from the regionmask library to the dataset.
    # ds = mask_land_with_regionmask(ds)
    
    time = pd.to_datetime(ds.time.values).strftime("%Y%m%d")
    init_time = pd.to_datetime(ds.init_time.values).strftime("%Y%m%d%H")
    if title is None:
        title = f"{init_time}_{time}"
    # 创建地图投影
    proj = ccrs.PlateCarree()
    # TODO 解决LambertConformal投影
    # proj = ccrs.LambertConformal(central_longitude=105, central_latitude=35, standard_parallels=(30, 50)) # for China
    
    # 创建图形并添加地图
    plt.figure(figsize=(10, 6), dpi=100)
    ax = plt.axes(projection=proj)


    # 创建颜色映射和norm
    if var == 'tp':
        colors = [
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
        levels = [-100, -80, -50, -20, 0, 20, 50, 100]  # 9 个分界点 -> 8 个颜色区间  panchen
        levels = [-80, -60, -40, -20, 0, 20, 40, 60, 80]  # 9 个分界点 -> 8 个颜色区间  panchen
        cblabel="%"
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, len(colors))
    elif var == 't2m':
        # 自定义颜色：紫-红-白-蓝
        cblabel="℃"
        colors = [
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
        levels = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]  # 11 个分界点 -> 10 个颜色区间
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, len(colors))
    else:
        raise ValueError("Invalid variable type. Please choose 'tp' or 't2m'.")

    # 绘制等高线图
    contour = ds.plot.contourf(ax=ax, levels=levels, cmap=cmap, norm=norm, add_colorbar=False)

    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='white')
    # ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='none')
    
    # 添加经纬网格线 Cannot label LambertConformal gridlines, Only PlateCarree gridlines are currently supported.
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
    # https://blog.csdn.net/qiuylulu/article/details/122004251
    # http://bbs.06climate.com/forum.php?mod=viewthread&tid=100563&highlight=%B0%D7%BB%AF
    # https://github.com/GaryBikini/ChinaAdminDivisonSHP.git  Country and rename to china
    # shpfile_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/WeatherVis/weathervis/utils/china_shp/china.shp"
    shpfile_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/WeatherVis/ChinaAdminDivisonSHP/1. Country/country.shp"
    # shapefile.Reader(shpfile)[0].record :['100000', '中华人民共和国']
    
    shp2clip(contour, ax, shpfile=shpfile_path, proj = proj, clabel = None, vcplot = False)

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ##  add provence line
    # geo_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/data/DEM/china-geospatial-data-UTF8/CN-border-La.gmt"
    # ax = add_provence(ax, geo_path)
    
    # add provence line
    shapefile_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/WeatherVis/ChinaAdminDivisonSHP/2. Province/province.shp"
    ax = plot_shapefile(ax, shapefile_path)
    
    # ax.set_extent([80, 134, 20, 48])
    # ax.set_extent([72, 136, 5, 53])
    ax.set_extent([70, 135, 15, 55]) # 中国经纬度边界
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

@time_decorator
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
    
import xarray as xr
@time_decorator
def plot_lambert(ds: xr.DataArray, save_path: str = "./tmp.png", levels=None, geo_path="", var="tp", title=None):
    """
    Plots a contour map of the given dataset `ds` and saves it to `fig_path`.

    Parameters:
    - ds (xarray.Dataset): The dataset containing the data to plot.
    - fig_path (str): The path where the figure will be saved.
    - levels (list, optional): The contour levels to use. If None, default levels are used.

    Returns:
    - ax (matplotlib.axes.Axes): The axes object containing the plot.
    """
    
    time = pd.to_datetime(ds.time.values).strftime("%Y%m%d")
    init_time = pd.to_datetime(ds.init_time.values).strftime("%Y%m%d%H")
    
    
    # Calculate latitude and longitude values
    lat = ds.lat.values
    lon = ds.lon.values
    
    # Create new latitude and longitude arrays with a step of 0.0001
    nlats, nlons = len(lat), len(lon)
    lats = np.linspace(np.min(lat), np.max(lat) + 0.0001, nlats)[:]#[::-1]
    lons = np.linspace(np.min(lon), np.max(lon) + 0.0001, nlons)[:]
    lons, lats = np.meshgrid(lons, lats)
    
    # Set the projection and create the figure and axis
    proj = ccrs.LambertConformal(central_longitude=105, central_latitude=35)  # China
    fig, ax = plt.subplots(1, 1, figsize=(10, 16), subplot_kw={"projection": proj})

    # 创建颜色映射和norm
    if var == 'tp':
        colors = [
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
        levels = [-100, -80, -50, -20, 0, 20, 50, 100]  # 9 个分界点 -> 8 个颜色区间  panchen
        levels = [-80, -60, -40, -20, 0, 20, 40, 60, 80]  # 9 个分界点 -> 8 个颜色区间  panchen
        cblabel="%"
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, len(colors))
    elif var == 't2m':
        # 自定义颜色：紫-红-白-蓝
        cblabel="℃"
        colors = [
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
        levels = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]  # 11 个分界点 -> 10 个颜色区间
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, len(colors))
    else:
        raise ValueError("Invalid variable type. Please choose 'tp' or 't2m'.")
    
    # Plot the contour map
    img = ax.contourf(lons, lats, ds[...], 
                      transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, levels=levels)  # must be ccrs.PlateCarree()
    
    # Add a colorbar  horizontal
    # cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.05, ax.get_position().width, 0.02])
    # plt.colorbar(img, cax=cax, orientation='horizontal')
    
    posn = ax.get_position()
    # 设置颜色条的位置和大小  vertical
    cax = plt.axes([posn.x1+0.04, posn.y0 + posn.height*1/16, 0.01, posn.height*7/8])  # [left, bottom, width, height]
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', cax=cax, extend='both')
    cbar.set_label(cblabel, rotation=0)  # 添加颜色条标签并设置为垂直方向
    
    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='white')
    # ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='none')
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      xlocs=[80, 90, 100, 110, 120, 130], ylocs=[20, 30, 40, 50],
                      draw_labels=False)
    
    gl.top_labels = False   # 不显示顶部标签
    gl.right_labels = False # 不显示右侧标签
    gl.left_labels = True   # 显示左侧标签
    gl.bottom_labels = True # 显示底部标签
    
    # mask china
    shpfile_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/WeatherVis/ChinaAdminDivisonSHP/1. Country/country.shp"
    # shapefile.Reader(shpfile)[0].record :['100000', '中华人民共和国']
    
    shp2clip(img, ax, shpfile=shpfile_path, proj = proj, clabel = None, vcplot = False)


    
    # add provence line
    geo_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/data/DEM/china-geospatial-data-UTF8/CN-border-La.gmt"
    add_provence(ax, geo_path)
    
    # # TODO: need fix 
    # shapefile_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/WeatherVis/ChinaAdminDivisonSHP/2. Province/province.shp"
    # ax = plot_shapefile(ax, shapefile_path)
        
        
    # Set the extent of the plot
    region = [70, 135, 15, 55]
    ax.set_extent(region, crs=ccrs.PlateCarree())
    
    # Save the plot
    if save_path is not None:
        save_path = os.path.join(save_path, init_time, f"{var}_{time}.png")
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=100, format='png')
        plt.close()
    else:
        return ax

