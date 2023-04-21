FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
RUN useradd -ms /bin/bash appuser

RUN apt update
RUN apt install git python3-pip -y && pip3 install -U pip

RUN git clone https://github.com/Gaspard-Bruno/visio-gptq.git
RUN chown -R appuser ./visio-gptq/*

WORKDIR /visio-gptq
ENTRYPOINT [ "./install.sh" ]

# RUN git config --global url.“https://***REMOVED***:@github.com/“.insteadOf “https://github.com/”
# RUN git clone https://github.com/Gaspard-Bruno/visio-gptq.git

# WORKDIR /visio-gptq
# COPY requirements.txt local_requirements.txt
# RUN pip install -U pip && pip install --no-cache-dir --upgrade -r local_requirements.txt

# RUN pip install -r requirements.txt

# COPY ./* .
# RUN chown -R appuser /visio-gptq

# ENTRYPOINT [ "./install.sh" ]
