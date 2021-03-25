FROM python:3.7

# Set the working directory
WORKDIR /computation

# Copy the current directory contents into the container
COPY . /computation

# Install any nddeeded packages specified in requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "entry.py"]

