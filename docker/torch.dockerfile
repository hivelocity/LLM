FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install python3 ptpython python3-pip git openssl
RUN pip install -U pip

RUN pip install torch
RUN pip install git+https://github.com/huggingface/transformers

RUN pip install datasets einops sentencepiece tokenizers protobuf
RUN pip install accelerate
