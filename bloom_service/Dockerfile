FROM python:3.10-buster


RUN apt-get update && apt-get install -y tzdata \
    && cp /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime \
    && echo "Asia/Ho_Chi_Minh" > /etc/timezone \
    && apt-get install -y ca-certificates jq

ADD . /BLOOM

RUN pip install -U pip \
    && pip install --no-cache-dir --upgrade -r /BLOOM/requirements.txt

RUN chmod +x /BLOOM/start_app.sh /BLOOM/load_model.py

WORKDIR /BLOOM
RUN /BLOOM/load_model.py   # Pre-loading model
RUN rm -rf offload/
CMD [ "./start_app.sh"]
