# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install tensorflow
RUN pip install transformers

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME Sexist

# Run when the container launches
CMD ["python", "main.py"]