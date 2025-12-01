#!/bin/bash

# echo "TODO: fill in the docker run command"
docker run -p 8080:8080 -e WANDBY_API_KEY=${WANDBY_API_KEY} ift6758/serving:latest