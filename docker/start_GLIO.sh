#!/bin/bash

docker container ls -a -f name=glio | grep glio$ > /dev/null

if [ $? == 0 ]
then
	docker container start glio
	docker exec -it glio /bin/bash

else
	xhost +
	docker run -it -e NVIDIA_DRIVER_CAPABILITIES="all" -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -v ./dataset:/root/dataset --gpus all --name glio docker_glio /bin/bash
fi
