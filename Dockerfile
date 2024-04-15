FROM ubuntu:20.04
RUN apt update -y && apt install -y python3 \
                                    python3-pip \
                                    wget
RUN apt install unzip
WORKDIR /app
COPY . .