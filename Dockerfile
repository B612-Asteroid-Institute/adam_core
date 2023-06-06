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

# Install pyoorb
RUN --mount=type=cache,target=/root/.cache/pip \
    export WHEEL_NAME="${OORB_VERSION}-cp310-cp310-manylinux_2_17_$(uname -m).manylinux2014_$(uname -m).whl" && \
    pip install "https://github.com/B612-Asteroid-Institute/oorb/releases/download/${OORB_TAG}/${WHEEL_NAME}"


# Upgrade pip to the latest version and install pre-commit
RUN pip install --upgrade pip pre-commit

# Install pre-commit hooks (before adam_core is installed to cache this step)
# Remove the .git directory after pre-commit is installed as adam_core's .git
# will be added to the container
RUN mkdir /code/
COPY .pre-commit-config.yaml /code/
WORKDIR /code/
RUN git init . \
	&& pre-commit install-hooks \
	&& rm -rf .git

# Install adam_core
ADD . /code/
RUN pip install -e .[tests]
