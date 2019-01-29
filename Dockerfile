FROM alpine:3.8

RUN apk add --no-cache python3 && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
    rm -r /root/.cache

RUN apk --no-cache --update-cache add gcc g++ gfortran python python3-dev wget freetype-dev libpng-dev openblas-dev

RUN apk add --no-cache \
            --allow-untrusted \
            --repository \
             http://dl-3.alpinelinux.org/alpine/edge/testing \
            hdf5 \
            hdf5-dev && \
    apk add --no-cache \
        build-base

RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN pip3 install --no-cache-dir flask

RUN pip3 install --no-cache-dir flask_compress

RUN pip3 install https://github.com/themech/alpine-python3-tensorflow/releases/download/alpine3.8-tensorflow1.10.1/tensorflow-1.10.1-cp36-cp36m-linux_x86_64.whl

RUN pip3 install --no-cache-dir numpy==1.16

RUN pip3 install --no-cache-dir pandas

RUN pip3 install --no-cache-dir scikit-learn

RUN pip3 install --no-cache-dir keras

COPY labelbinarizer.pkl /

COPY lrensemble_model.pkl /

COPY lstm_model.h5 /

COPY valid_chars.pkl /

COPY TLDlist.txt /

COPY app.py /

EXPOSE 80

CMD ["python3", "app.py"]
