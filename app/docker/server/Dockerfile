FROM python:3.9.17-buster

# Set the working directory
WORKDIR /app

RUN apt-get update

# Install the required dependencies
RUN apt-get install -y libhdf5-dev

COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

CMD ["python", "-m", "src.server.server_handler"]