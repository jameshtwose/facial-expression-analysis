# facial-expression-analysis
A set of scripts looking into analyses based on facial expression feature extraction.

## Pre-requisites
- understanding [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units#extraction-from-images-and-extraction-from-videos)
- having a video file to analyse
- having a docker image of openface (see below)
- python for running the analysis scripts

## How to create data (docker)
- `docker run -it --rm algebr/openface:latest` install the docker image of open face (this should also open a bash shell in the root of the docker image)
- `cd /home/openface-build/build/bin/` change directory to the openface binaries directory
- in a different terminal run:
  - `docker ps` to get the container id
  - `docker cp <video_file>.mp4 <container_id>:/home/openface-build/build/bin/` to copy a video file to the docker image
- `./FeatureExtraction -f <video_file>.mp4` to run the feature extraction on the video file
- in a different terminal run:
  - `docker cp <container_id>:/home/openface-build/build/bin/processed/<video_file>.csv .` to copy the output file to the host machine

## Analyse created data
- `pip install -r requirements.txt` to install the python dependencies
- `streamlit run app.py` to run the streamlit app
