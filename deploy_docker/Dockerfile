FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy1.3.0

#COPY ./ /root/workspace/mmsegmentation/
#ENV FORCE_CUDA="1"
#RUN pip install -r /root/workspace/mmsegmentation/requirements.txt
#RUN pip install --no-cache-dir -e /root/workspace/mmsegmentation

# previous method of local installation fails so fallback to the official method
RUN mim install mmsegmentation
RUN python3 -m pip install ftfy regex

RUN git clone https://github.com/logivations/mmsegmentation.git /code/mmsegmentation
