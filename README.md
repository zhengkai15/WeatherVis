# WeatherVis

### CHANGE LOG
- 
- 支持绘制中国兰勃脱投影（大陆mask、中国之外mask） 
- 支持绘制中国温度和降水距平场填色图（添加省界）
- 增加风羽时间图


WeatherVis 是一个用于气象数据可视化的 Python 工具库，支持多种气象图形的绘制，包括：
 - 时间序列图
 - 填色地图
 - 风场填色图

### 功能概述
1. 时间序列图
 - 功能描述：展示气象变量（温度、降水量、风速等）随时间的变化趋势。
 - 主要特性：
 - 支持多变量对比绘制。
 - 可自定义线型、颜色、标记等样式。
 - 添加趋势线、显著标记和时间区间填充等功能。

2. 填色地图（Filled Contour Map）
 - 功能描述：展示气象变量（温度、降水、湿度等）的空间分布。
 - 主要特性：
 - 支持自定义色卡（如降水量：褐色到绿色）。
 - 可叠加地理信息（海岸线、行政边界、河流等）。
 - 支持不同区域（子域）数据可视化，灵活设置经纬度范围。

3. 风场填色图（Wind Field Coloring）
 - 功能描述：展示风场数据的空间分布，包括风向和风速。
 - 主要特性：
 - 使用填色地图表示风速大小，流线或箭头表示风向。
 - 支持自定义色卡和透明度。
 - 叠加地理特征和标签标注，清晰展示风场分布。

### 环境依赖
本项目基于以下 Python 库：
 - python >= 3.8
 - xarray：处理 NetCDF 等气象数据格式
 - matplotlib：用于数据可视化
 - cartopy：叠加地理信息，支持地图绘制
 - numpy：用于数值计算
 - pandas（可选）：时间序列数据处理

### 安装依赖
运行以下命令安装所需依赖：
```bash
# 配置shp文件
git clone https://github.com/GaryBikini/ChinaAdminDivisonSHP.git
pip install xarray matplotlib cartopy numpy pandas
```

### 使用说明
 - 支持格式：NetCDF (.nc)、CSV、GRIB 等标准气象数据格式。
 - 示例数据：建议数据包含时间、经纬度及对应的变量（如 temperature、precipitation、wind_u 和 wind_v 等）。



### 贡献指南
 - 提交 Issue 提出改进建议或发现的 Bug。
 - Fork 本仓库并提交 Pull Request，分享您的新功能。

### 许可证
本项目遵循 MIT License 开源许可协议。

### 致谢
感谢所有气象数据科学家和开发者为社区做出的贡献！
这样一个仓库命名和 README 能够简洁清晰地展示项目的目的和功能，同时易于扩展和维护。
