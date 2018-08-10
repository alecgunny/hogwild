FROM nvcr.io/nvidia/tensorflow:18.07-py3

WORKDIR /workspace/

COPY Hogwild.py run.sh /workspace/

ENTRYPOINT ["./run.sh"]
