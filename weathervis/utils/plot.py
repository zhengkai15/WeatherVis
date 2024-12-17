import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

def add_provence(ax,geo_path):
    with open(geo_path) as src:
        context = ''.join([l for l in src if not l.startswith('#')])
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
    for line in borders:
        ax.plot(line[0::2], line[1::2], '-', lw=0.5, color='k',
                transform=ccrs.PlateCarree()) 
    return ax


def plot(ds, save_path="./",  level=None, geo_path="", time=None, init_time=None, title="", var="tp"):
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
        levels1 = [-100, -80, -50, -20, 0, 20, 50, 100, 200]  # 9 个分界点 -> 8 个颜色区间
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
        levels2 = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]  # 11 个分界点 -> 10 个颜色区间
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
    ax.add_feature(cfeature.OCEAN, facecolor='white')
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

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax = add_provence(ax, geo_path)
    # ax.set_extent([80, 134, 20, 48])
    ax.set_extent([70, 140, 0, 53])
    # plt.savefig(save_path)
