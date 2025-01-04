# FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04 
# for driver 560.xx to be released when it's less buggy.
FROM python:3.10

RUN apt update && apt install python3-pip git python3-venv -y
RUN apt install -y libgl1-mesa-glx libosmesa6

# create virtual environment, the correct way https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# install jax with cuda support
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade jax[cuda12]==0.4.38 flax ott-jax
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "tasks/benchmark.py"]