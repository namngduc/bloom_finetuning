FROM python:3.9-buster


RUN apt-get update && apt-get install -y tzdata \
    && cp /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime \
    && echo "Asia/Ho_Chi_Minh" > /etc/timezone \
    && apt-get install -y ca-certificates jq

ADD . /BLOOM

RUN pip install -U pip \
    && pip install --no-cache-dir --upgrade -r /BLOOM/requirements.txt

RUN chmod +x /BLOOM/app.py /BLOOM/load_model.py

WORKDIR /BLOOM
RUN /BLOOM/load_model.py   # Pre-loading model
CMD [ "python", "-u", "/BLOOM/app.py"]
