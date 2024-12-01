# DEM Enhancement Using Lidar

## Running the Tool Using Docker

Follow these steps to build and run the tool with Docker:

1. **Build the Docker Image**:
   - Navigate to the directory containing the Dockerfile and other necessary files:
     ```bash
     cd path/to/repo/files
     ```
   - Build the Docker image with the following command:
     ```bash
     docker build -t pdal_for_lidar .
     ```

2. **Run the Docker Container**:
   - Start a Docker container using the created image:
     ```bash
     docker run -it -p 8501:8501 -v .:/app pdal_for_lidar /bin/bash
     ```

3. **Launch the Streamlit App**:
   - Inside the container, run the Streamlit app:
     ```bash
     streamlit run app.py
     ```

4. **View the App**:
   - Open your browser and navigate to:
     [http://localhost:8501](http://localhost:8501)


