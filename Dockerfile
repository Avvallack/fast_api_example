FROM python:3.9.6-buster

RUN apt-get update -y && apt-get upgrade -y

WORKDIR /home/fast_api_worker
RUN mkdir /home/fast_api_worker/data
COPY /src /home/fast_api_worker/src
COPY main.py /home/fast_api_worker
COPY requirements.txt /home/fast_api_worker
RUN pip install -r requirements.txt
RUN python3 $(pwd)/src/get_data.py
RUN python3 $(pwd)/src/train_model.py

EXPOSE 8000
CMD uvicorn main:app --host "0.0.0.0" --port "8000"