FROM python:3

RUN echo "$USER ALL = NOPASSWD: ALL" >> /etc/sudoers && \

    apt-get update && apt-get install -y \
        gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev

COPY coinhsl-2019.05.21 /tmp/coinhsl-2019.05.21/ 
ADD test_cyipopt.py /
ADD get_ipopt.sh /tmp/
ADD get_cyipopt.sh /tmp/
RUN bash /tmp/get_ipopt.sh
ENV IPOPTPATH /opt/ipopt
ENV PKG_CONFIG_PATH $PKG_CONFIG_PATH:$IPOPTPATH/lib/pkgconfig
ENV PATH $PATH:$IPOPTPATH/bin
RUN bash --login /tmp/get_cyipopt.sh
RUN rm /tmp/get_ipopt.sh /tmp/get_cyipopt.sh

CMD [ "python", "./test_cyipopt.py" ]