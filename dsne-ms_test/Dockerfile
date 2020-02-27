#FROM coinstac/coinstac-base
FROM coinstac/coinstac-base-python-stream 

# Set the working directory
WORKDIR /computation

COPY requirements.txt /computation

RUN pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . /computation

# Install any needed packages specified in requirements.txt
# RUN pip install -r requirements.txt
