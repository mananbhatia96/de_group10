FROM python:3.10.7-slim
WORKDIR /pipelines/component/src
ADD src/component.py /pipelines/component/src/component.py
ADD requirements.txt /pipelines/component/src/requirements.txt
RUN pip install -r requirements.txt
