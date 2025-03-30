FROM python:slim

RUN apt update && apt install make
RUN pip install --upgrade pip && pip install uv
