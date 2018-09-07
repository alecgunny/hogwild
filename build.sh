#!/bin/bash
docker build -t $USER/hogwild:gpu --build-arg tag=latest-gpu .
docker build -t $USER/hogwild:cpu --build-arg tag=latest .
