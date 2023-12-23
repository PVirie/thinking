FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

RUN apt update && apt install python3-pip git -y
RUN pip3 install --upgrade pip

RUN pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt 
CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "tasks/main.py"]