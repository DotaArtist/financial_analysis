FROM base-registry.zhonganinfo.com/nuanwa/python3_base:0923

#RUN apt-get update \
#  && apt install -y python2.7

RUN /usr/bin/pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytz==2017.2 \
  && /usr/bin/pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple PyMySQL==0.7.11 \
  && /usr/bin/pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple APScheduler==3.3.1

ADD . /nuanwa-text-summarization-model/
WORKDIR /nuanwa-text-summarization-model/prediction/

EXPOSE 8001

#ENTRYPOINT ["python2.7", "standalone.py"]
ENTRYPOINT ["/usr/bin/python3", "main.py"]
