FROM python:3.12.3
# FROM ubuntu:24.04

COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY cache /app/cache
COPY data_explorer.py /app/data_explorer.py

WORKDIR /app

# RUN apt-get update
# RUN apt-get install -y kmod kbd
RUN pip install -r requirements.txt


CMD ["python", "data_explorer.py"]
