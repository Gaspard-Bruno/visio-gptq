FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

RUN useradd -ms /bin/bash app
USER app
WORKDIR /home/app

USER root
RUN apt update && apt install -y git python3-pip
RUN git clone --recursive https://github.com/Gaspard-Bruno/visio-gptq.git

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN chown -R app /home/app/visio-gptq
WORKDIR /home/app/visio-gptq

USER app
RUN chmod +x install.sh

CMD ["./install.sh"]