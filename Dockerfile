FROM nvcr.io/nvidia/pytorch:22.07-py3
WORKDIR /workspace
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt
