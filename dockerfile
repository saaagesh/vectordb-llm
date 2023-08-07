FROM python:3.10-bullseye

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY main.py main.py
COPY Procfile Procfile

RUN echo "export PATH=$PATH:/usr/local/bin/" >> ~/.bashrc

EXPOSE 8080
 
CMD [ "python", "./main.py" ]