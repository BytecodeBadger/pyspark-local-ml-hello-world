# Use a base image with Spark and Python
FROM apache/spark:latest

USER root
RUN pip install numpy

# Set the working directory in the container
WORKDIR /app

# Copy the PySpark code into the container
COPY linear_regression.py /app/linear_regression.py

# Entry point
CMD ["/opt/spark/bin/spark-submit", "linear_regression.py"]