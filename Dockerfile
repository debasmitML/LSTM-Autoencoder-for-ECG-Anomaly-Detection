
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04


WORKDIR /main



RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install python3.8 -y
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip


COPY . /main



RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip3 install -r requirements.txt


ENTRYPOINT ["python3"]

CMD ["run.py"]