# Use Ubuntu 20.04 as base image
FROM ubuntu:20.04

# Disable interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt update && apt install -y python3 python3-pip

RUN apt update && apt install -y libblas-dev liblapack-dev gfortran build-essential



# Set working directory
WORKDIR /app

# Copy the Python script into the container
COPY . .

#RUN pip install --no-binary :all: quadprog

RUN pip3 install -r requirements.txt

RUN pip uninstall quadprog -y
RUN pip install --no-binary :all: quadprog

# Set the default command to run your script
CMD ["python3", "global_racetrajectory_optimization/main_globaltraj.py"]
