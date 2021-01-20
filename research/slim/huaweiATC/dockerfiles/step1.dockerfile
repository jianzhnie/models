FROM ubuntu:18.04
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update &&  apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev openssl libssl-dev libffi-dev unzip pciutils net-tools
RUN apt-get install -y sudo iproute2

# Install vim、 wget
RUN apt-get install -y vim 
RUN apt-get install -y wget
RUN apt-get install -y libblas-dev gfortran libblas3 libopenblas-dev