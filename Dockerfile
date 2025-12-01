# 使用您本地已存在的昇腾PyTorch基础镜像
FROM pytorch_ascend:2.1.0

# 设置容器内的工作目录
WORKDIR /app/Salience-DETR

# 将当前目录下的requirements.txt文件复制到镜像中
COPY requirements.txt .

# （可选但推荐）设置国内PyPI镜像源以加速下载
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装项目所需的Python依赖包
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量，确保项目代码能正确找到Python路径
ENV PYTHONPATH=/app/Salience-DETR:$PYTHONPATH

# 设置默认的命令行入口
CMD ["/bin/bash"]
