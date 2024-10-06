FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the /app directory in the container
COPY . /app


RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["gunicorn"  , "-b", "0.0.0.0:8888", "app:app"]