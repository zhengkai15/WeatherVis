from setuptools import setup, find_packages

setup(
    name="weathervis",               # 包的名字
    version="0.1.0",              # 版本号
    author="kaizheng",           # 作者名字
    author_email="156252108@qq.com",  # 作者邮件
    description="WeatherVis 是一个用于气象数据可视化的 Python 工具库，支持多种气象图形的绘制。",  # 简要描述
    long_description=open('README.md').read(),  # 从 README.md 文件读取详细描述
    long_description_content_type="text/markdown",  # markdown 格式
    url="https://github.com/zhengkai15/WeatherVis",  # 项目 URL
    packages=find_packages(),  # 自动查找所有子包
    install_requires=[          # 外部依赖
        # 列出需要安装的依赖包
        "numpy",
        "pandas",
    ],
    classifiers=[               # 额外分类信息
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',    # 需要的 Python 版本
)