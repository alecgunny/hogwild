ARG tag=latest-gpu

FROM tensorflow/tensorflow:$tag

WORKDIR /workspace/

ENTRYPOINT ["python", "Hogwild.py"]
