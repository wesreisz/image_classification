FROM python:3.7.5-slim

RUN apt-get clean \
    && apt-get -y update

RUN apt-get -y install nginx \
    && apt-get -y install python3-dev \
    && apt-get -y install build-essential \
    && apt-get -y install curl

COPY flask_app/ /apps/flask_app
WORKDIR /apps/flask_app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

EXPOSE 5000
COPY nginx/nginx.conf /etc/nginx

RUN mkdir -p /opt/section/
COPY proxy/xception_weights_tf_dim_ordering_tf_kernels.h5 /opt/section/
COPY run.sh /opt/section/
RUN chmod +x /opt/section/run.sh
CMD ["/opt/section/run.sh"]
