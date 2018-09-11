FROM ubuntu:16.04

FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update | apt-get upgrade
RUN apt-get install -y \
    curl \
    git \
    vim \
    wget

# Python3
RUN apt-get install -y python3-dev python3 python3-pip python3-tk
RUN pip3 install pip --upgrade

# TensorFlow
RUN pip3 install tensorflow-gpu

# Sonotano raiburari
RUN pip3 install numpy matplotlib scipy scikit-image tqdm moviepy

# jupyter
RUN pip3 install jupyter

# HDF5
RUN pip3 install h5py

# Pillow
RUN pip3 install pillow

# Enables X11 sharing and creates user home directory
ENV USER_NAME pbt_test
ENV HOME_DIR /home/$USER_NAME
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1013
ENV HOST_GID 2000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    mkdir /etc/sudoers.d/ && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0666 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

RUN mkdir population_based_training_tensorflow
RUN mkdir .jupyter
COPY jupyter_notebook_config.py ./.jupyter

CMD /bin/bash
