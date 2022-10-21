FROM python:3.8
RUN apt-get update
RUN mkdir -p /etc/docker/certs.d/looker.ovh
ADD cert/looker.ovh /etc/docker/certs.d/looker.ovh
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -U pip && pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./ /app/
CMD uvicorn app:server --host 0.0.0.0 --port 80
