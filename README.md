setup:
docker build -t ganns-ades .
docker run -v ~/ganns-ades:/home/pfcarvalho/ganns-ades --gpus all -it -d ganns-ades   

delete image:
docker rmi ganns-ades

regular:
docker ps
docker attach 17af4a19f780

ctrl p+q to leave