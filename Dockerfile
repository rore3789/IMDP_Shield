
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

#ARG USER_ID
#ARG GROUP_ID
#
#RUN addgroup --gid $GROUP_ID user
#RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
#RUN chown -R user:user /var/lib/apt/lists/
#USER user

# update and install basic tools and python
RUN apt-get update
RUN apt-get install -y software-properties-common iputils-ping curl wget build-essential cmake git libopenblas-dev liblapack-dev
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
RUN rm -rf /var/lib/apt/lists/*

# setup python 3.10 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# update and install julia
RUN apt-get update && apt-get install -y wget
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.10.4-linux-x86_64.tar.gz
RUN mv julia-1.10.4 /opt/
RUN ln -s /opt/julia-1.10.4/bin/julia /usr/local/bin/julia
RUN rm julia-1.10.4-linux-x86_64.tar.gz

# install cuda
RUN apt-get update && apt-get install -y --no-install-recommends nvidia-cuda-toolkit

# ensure pip works with a manual install
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py

COPY ./src /usr/src/SHIELD
WORKDIR /usr/src/SHIELD

# add python packages
RUN python3 -m pip install --upgrade pip --no-cache-dir
RUN python3 -m pip install cvxpy pertdist juliacall gpytorch pyyaml
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN cd auto_LiRPA && python3 -m pip install .

# add julia packages, it doesn't like doing too many at once
RUN julia -e 'using Pkg; Pkg.update()'
RUN julia -e 'using Pkg; Pkg.add(["SparseArrays", "Distributed", "SharedArrays", "PyCall", "Distributions"])'
RUN julia -e 'using Pkg; Pkg.add(["Random", "ProgressBars", "NetCDF", "SpecialFunctions", "MAT"])'
RUN julia -e 'using Pkg; Pkg.add(["JuMP", "Ipopt", "Plots", "ColorSchemes", "Printf"])'
RUN julia -e 'using Pkg; Pkg.add(["NPZ", "Flux", "JLD", "LinearAlgebra", "YAML"])'
RUN julia -e 'using Pkg; Pkg.add(name="IntervalMDP", version="0.1.0")'
RUN julia -e 'using Pkg; Pkg.add(url="https://github.com/aria-systems-group/PosteriorBounds.jl")'

ENTRYPOINT ["bash"]