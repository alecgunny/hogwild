FROM nvcr.io/nvidia/tensorflow:18.07-py3

WORKDIR /workspace/

COPY Hogwild.py /workspace/

ENTRYPOINT ["python", "Hogwild.py"]
