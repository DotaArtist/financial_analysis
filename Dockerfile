FROM swr.cn-east-3.myhuaweicloud.com/dota_artist/my_ubuntu:with_python36

RUN /usr/local/bin/pip3 install jqdatasdk==1.8.0 \
  && /usr/local/bin/pip3 install scikit_learn==0.21.2 -i https://pypi.tuna.tsinghua.edu.cn/simple \
  && /usr/local/bin/pip3 install pdfminer.six==20201018 \
  && /usr/local/bin/pip3 install argparse==1.1 \
  && /usr/local/bin/pip3 install logging==0.5.1.2 \
  && /usr/local/bin/pip3 install h5py==3.1.0 \
  && /usr/local/bin/pip3 install vitables
