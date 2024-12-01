# Use the official pdal image
FROM pdal/pdal

WORKDIR /app  

# Copy requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .   

