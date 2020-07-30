
docker run -ti \
--gpus=all \
--ipc=host \
-v $HOME/.datasets:/root/.datasets \
-v $PWD:/workspace \
-p 8097:8097 \
--name feratt-run feratt:latest \
/bin/bash