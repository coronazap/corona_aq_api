FROM python:3.6
ADD . /app 
WORKDIR /app 
RUN pip3 install -r requirements.txt
RUN python download.py
CMD python /app/app.py 