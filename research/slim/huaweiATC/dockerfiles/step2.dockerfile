FROM ubuntu:18.04
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update &&  apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev libsqlite3-dev openssl libssl-dev libffi-dev unzip pciutils net-tools
RUN apt-get install -y sudo iproute2

# Install vim、 wget
RUN apt-get install -y vim 
RUN apt-get install -y wget
RUN apt-get install -y libblas-dev gfortran libblas3 libopenblas-dev

# Install python3.7.5
RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
# For faster build time, modify the -j flag according to your processor. If you do not know the number of cores your processor, you can find it by typing nproc
RUN tar -xvf Python-3.7.5.tgz
WORKDIR /Python-3.7.5
RUN ./configure --prefix=/usr/local/python3.7.5 --enable-shared
RUN make -j4 && make install

# enviroment for ATC
RUN ./configure --prefix=/usr/local/python3.7.5 --enable-shared
RUN cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 /usr/lib
RUN ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7
RUN ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7
RUN ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7.5
RUN ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7.5


# update pip 
RUN pip3.7.5 config set global.index-url https://mirrors.aliyun.com/pypi/simple
# Install py-tools
# RUN pip3.7.5 install --upgrade pip
# RUN pip3.7.5 install psutil \
#     pip3.7.5 install decorator \
#     pip3.7.5 install numpy \ 
#     pip3.7.5 install protobuf==3.11.3 \ 
#     pip3.7.5 install scipy \
#     pip3.7.5 install sympy \
#     pip3.7.5 install cffi \
#     pip3.7.5 install grpcio \
#     pip3.7.5 install grpcio-tools \
#     pip3.7.5 install requests \ 
#     pip3.7.5 install attrs \

## Ascend-Toolkit
COPY /home/robin/software/Ascend-Toolkit-20.0.RC1-x86_64-linux_gcc7.3.0.run .
RUN ./Ascend-Toolkit-20.0.RC1-x86_64-linux_gcc7.3.0.run


# # control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
ENV PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/bin:$PATH
ENV PYTHONPATH=/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/python/site-packages/te:/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/python/site-packages/topi:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/lib64:$LD_LIBRARY_PATH
ENV ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/20.1.rc1/opp

