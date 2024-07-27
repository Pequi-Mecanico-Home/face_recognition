FROM dustynv/ros:foxy-ros-base-l4t-r35.4.1

# Remove opencv - problemas com lib
RUN apt-get purge -y '*opencv*' && apt autoremove -y

RUN apt-get update && apt-get upgrade -y

# BUILDING OPENCV4.8
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git unzip pkg-config zlib1g-dev \
    libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libglew-dev \
    libgtk2.0-dev libgtk-3-dev libcanberra-gtk* \
    python3-dev python3-numpy python3-pip \
    libxvidcore-dev libx264-dev libgtk-3-dev \
    libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev \
    gstreamer1.0-tools libv4l-dev v4l-utils qv4l2 \
    libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
    libavresample-dev libvorbis-dev libxine2-dev libtesseract-dev \
    libfaac-dev libmp3lame-dev libtheora-dev libpostproc-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libopenblas-dev libatlas-base-dev libblas-dev \
    liblapack-dev liblapacke-dev libeigen3-dev gfortran \
    libhdf5-dev protobuf-compiler \
    libprotobuf-dev libgoogle-glog-dev libgflags-dev

# CUDA location
RUN echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf && ldconfig

# Download and build OpenCV
RUN cd ~ && \
    git clone --depth=1 https://github.com/opencv/opencv.git && \
    git clone --depth=1 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -D WITH_OPENCL=OFF \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=5.3 \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_NEON=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENMP=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_PROTOBUF=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF .. && \
    FREE_MEM="$(free -m | awk '/^Swap/ {print $2}')" && \
    NO_JOB=1 && \
    if [[ "$FREE_MEM" -gt "5500" ]]; then NO_JOB=4; fi && \
    make -j ${NO_JOB}

RUN rm -rf /usr/include/opencv4/opencv2 && \
    cd ~/opencv/build && \
    make install && \
    make clean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install -y \
        ros-foxy-realsense2-camera \
        ros-foxy-rviz2 \
        git

# Instalando ultralytics e removendo o opencv instalado como dependência
RUN pip install ultralytics

RUN pip uninstall opencv-python -y

# Instalando versão correta do pytorch(1.12) - de acordo com o Jetpack 5.0
# Para mais informações: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

ARG TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

RUN apt-get update && apt-get install -y \
        libopenblas-base \
        libomp-dev

RUN python3 -m pip install --upgrade pip && pip3 install --no-cache ${TORCH_INSTALL}

# Arrumando versão torchvision
ENV BUILD_VERSION=0.13.0

RUN apt-get update && apt-get install -y \
        libjpeg-dev \
        zlib1g-dev \
        libpython3-dev \
        libopenblas-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev

# Clone torchvision repository
RUN git clone --branch v0.13.0 https://github.com/pytorch/vision /opt/torchvision

# Install torchvision
RUN cd /opt/torchvision && python3 setup.py install --user

# Instalando dependências para rodar com Tensorrt
RUN pip install \
        onnx \
        onnxsim \
        onnxruntime

WORKDIR /dev_ws

RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash; colcon build --symlink-install' && \ 
    echo "source ${ROS_ROOT}/install/setup.bash" >> /root/.bashrc && \ 
    echo "source install/setup.bash" >> /root/.bashrc && \
    echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc

# Comandos adicionais
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev \
    liblapack-dev libblas-dev gfortran python3-pip

RUN python3 -m pip install --upgrade pip && \
    pip3 install -U testresources setuptools==65.5.0 && \
    pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 \
    gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.7.0 && \
    pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06 && \
    pip install h5py==3.10.0

RUN pip3 install opencv-python==4.9.0.80 ||true

RUN pip install -i http://jetson.webredirect.org/root/pypi deepface==0.0.91 --trusted-host jetson.webredirect.org || true

RUN pip install --upgrade numpy

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

ENTRYPOINT [ "bash" ]
