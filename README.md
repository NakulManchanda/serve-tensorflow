# serve-tensorflow
server pre-trained tensorflow model

# serve tutorial
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple


# setup

## create & activate virtualenv
```
virtualenv ~/.venv
echo "source virtualenv ~/.venv/bin/activate" >> ~/.bashrc
source virtualenv ~/.venv/bin/activate
```

## build & run
```
make install
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip install -U numpy
python main.py
```