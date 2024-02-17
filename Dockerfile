FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN . /root/.bashrc && \
    conda env create -f environment.yml && \
    conda init bash && \
    conda activate classifier

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY preload.py preload.py
RUN python preload.py
COPY . .

EXPOSE 8081

ENTRYPOINT ["python", "app.py"]