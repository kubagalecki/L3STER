FROM    ubuntu:latest

RUN         apt update \
        &&  apt install -y \
                gcc \
                g++ \
                make \
                cmake \
                git \
                wget \
                openmpi \
        &&  git clone https://github.com/catchorg/Catch2.git \
        &&  cd Catch2 \
        &&  cmake -Bbuild -H. -DBUILD_TESTING=OFF \
        &&  cmake --build build/ --target install
