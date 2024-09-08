FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

RUN apt update && apt install python3-pip git python3-venv -y

# added CUDNN to path
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

# create virtual environment, the correct way https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# install jax with cuda support
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade "jax[cuda12]"
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "tasks/benchmark.py"]