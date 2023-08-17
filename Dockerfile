FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get -y install screen
RUN echo "shell /bin/bash" > ~/.screenrc
RUN apt-get install git -y
RUN apt-get install wget -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install curl -y
RUN apt-get install libsdl-image1.2-de -y
RUN apt-get install yasm

RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

RUN pip install matplotlib
RUN pip install av
RUN pip install notebook
RUN pip install jupyterlab
RUN pip install ipywidgets
RUN pip install pytorch-lightning
RUN pip install lightning-bolts
RUN pip install scikit-image
RUN pip install opencv-python
RUN pip install pytorch-msssim
RUN pip install lpips
RUN pip install torch-fidelity
RUN pip install torch-ema
RUN pip install yacs
RUN pip install Cython

RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.4
RUN pip install opencv-python
RUN pip install numpy
RUN pip install meshzoo
RUN pip install pillow 
RUN pip install tensorboard
RUN pip install tqdm
RUN pip install trimesh
#RUN pip install git+https://github.com/NVlabs/nvdiffrast.git

EXPOSE 8888
EXPOSE 6006

COPY . /app/
WORKDIR /app
#RUN pip install --no-index --find-links ext/pyremesh pyremesh

ENTRYPOINT ["tail", "-f", "/dev/null"]
