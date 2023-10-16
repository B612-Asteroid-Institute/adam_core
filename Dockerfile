FROM ubuntu:jammy

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get upgrade -y \
    && apt-get install libcurl4-openssl-dev libssl-dev git curl unzip python3.11 pip -y

# Install pyoorb from B612's fork, which includes a patch to handle fortran errors better
ENV OORB_TAG=v1.2.1a1.dev2
ENV OORB_VERSION="pyoorb-1.2.1a1.dev2+66b7753.dirty"

# Install oorb data
RUN curl -fL -o /tmp/oorb_data.zip \
    "https://github.com/B612-Asteroid-Institute/oorb/releases/download/${OORB_TAG}/oorb_data.zip"
RUN unzip -d /opt/oorb_data /tmp/oorb_data.zip
ENV OORB_DATA=/opt/oorb_data

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install hatch
RUN pip install hatch

# Install adam_core
ADD . /code/
WORKDIR /code

RUN apt-get update -y && apt-get install -y gfortran liblapack-dev

# Force installation of deps by running a command in the 'dev' environment
RUN hatch -v run dev:true



