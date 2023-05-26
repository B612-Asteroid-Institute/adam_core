FROM ubuntu:jammy

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install libcurl4-openssl-dev libssl-dev git python3.11 pip -y

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
