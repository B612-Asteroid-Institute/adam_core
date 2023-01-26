FROM ubuntu:jammy

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install libcurl4-openssl-dev libssl-dev git python3 pip -y

RUN mkdir /code/
ADD . /code/
WORKDIR /code/
RUN pip install -e .[tests]