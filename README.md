# Lidar-Processor

## Runnig the tool using Docker

1- Build a Docker image:
   - Change directory cd) to the directory containing the files, including Dockerfile
   - run this command:     docker build -t pdal_for_lidar .
2- Run a docker container using the image
docker run -it -p 8501:8501 -v .:/app pdal_for_lidar /bin/bash

3- run the strealit app by:
streamlit run app.py

4- view the app on browser on http://localhost:8501/
