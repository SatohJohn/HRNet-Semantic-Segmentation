FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
RUN mkdir app && apt-get update && apt-get -y upgrade && apt-get install -y libgl1-mesa-dev libopencv-dev && pip install --upgrade pip
WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
