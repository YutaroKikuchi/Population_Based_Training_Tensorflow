#!/bin/bash

if [[ ! -z  `which nvidia-docker`  ]]
then
    DOCKER_CMD=nvidia-docker
elif [[ ! -z  `which docker`  ]]
then
    echo "WARNING: nvidia-docker not found. Nvidia drivers may not work." >&2
    DOCKER_CMD=docker
else
     echo "ERROR: docker or nvidia-docker not found. Aborting." >&2
    exit 1
fi


SCRIPT_DIR=$(pwd)
echo $SCRIPT_DIR
${DOCKER_CMD} run -ti --net=host --privileged -e DISPLAY=${DISPLAY} -v ${SCRIPT_DIR}:/home/pbt_test/population_based_training_tensorflow --rm --name population_based_train_tensorflow population_based_training_tensorflow
