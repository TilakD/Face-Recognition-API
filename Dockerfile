FROM nvidia/cuda:9.0-devel-ubuntu16.04

# Set proxies (if needed)
#ENV http_proxy ................
#ENV https_proxy ................

# Pick up some TF dependencies
ENV CUDNN_VERSION=7.4.2.24-1+cuda9.0
ENV NCCL_VERSION=2.4.2-1+cuda9.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=${CUDNN_VERSION} \
        libcudnn7-dev=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libsm6 \
        libzmq3-dev \
        ffmpeg \
        imagemagick \
        pkg-config \
        python3-tk \
        software-properties-common \
        unzip \
        vim \
        python3-openssl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}


RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} install --upgrade \
    pip \
    setuptools

ARG TF_PACKAGE=tensorflow-gpu==1.9.0
RUN ${PIP} install ${TF_PACKAGE}


RUN apt-get install -y python3
RUN apt-get install -y vim
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender1
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-openssl

RUN ${PIP} install \
    dlib \
    imutils \
    joblib \
    keras \
    matplotlib \
    numpy \
    opencv-contrib-python \
    pandas \
    scipy \
    scikit-learn \ 
    scikit-image \
    tqdm \
    flask \
    Flask-Cors \
    grpcio \
    grpcio-tools \
    tensorflow-serving-api \
    sklearn

WORKDIR /app

COPY app/* ./app/

ENTRYPOINT [ "python3" ]

EXPOSE 8998
CMD ["app/Stub_FR.py"]
