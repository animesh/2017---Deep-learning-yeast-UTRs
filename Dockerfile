FROM jupyter/base-notebook

ENV NB_UID=999 \
    NB_GID=999

RUN conda install theano=0.9.0 keras==1.2.2 networkx==1.11 pandas seaborn pygpu mkl-service
# Setup Jupyter Config and notebooks directory
ENV NOTEBOOKS_CONFIG_DIR /etc/jupyter
COPY jupyter_notebook_config.py $NOTEBOOKS_CONFIG_DIR/
USER root
RUN mkdir -p /data /data/notebooks
RUN fix-permissions /data /opt/conda
RUN apt update && apt install -y nano vim g++ liblapack-dev libopenblas-dev python-dev graphviz \
	&& apt autoclean && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Install cuDNN library
RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION 8.0.61

ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-nvgraph-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-8-0=8.0.61.2-1 \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION \
        cuda-npp-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-driver-dev-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

ADD libcudnn7*.deb /tmp/
RUN dpkg -i /tmp/libcudnn7*.deb && \
	rm -f /tmp/libcudnn7*.deb

RUN echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/cuda-8.0/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin/:/usr/local/cuda-8.0/bin/:${PATH}
RUN pip install hyperopt==0.1
#RUN pip install https://github.com/lebedov/scikit-cuda.git#egg=scikit-cuda
RUN ln -s /usr/local/cuda-8.0/include/* /usr/include/
WORKDIR /data/notebooks
RUN echo "[global]\ndevice=cuda\nfloatX=float32\n[gpuarray]\npreallocate=1\n[blas]\nldflags = -lopenblas\n" > $HOME/.theanorc
RUN fix-permissions $HOME/.theanorc
COPY start.sh /usr/local/bin/