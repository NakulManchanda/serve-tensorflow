# serve-tensorflow
server pre-trained tensorflow model

# serve tutorial
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple


# setup

## create & activate virtualenv
```
virtualenv ~/.venv
echo "source virtualenv ~/.venv/bin/activate" >> ~/.bashrc
echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" >> ~/.bashrc
echo "alias pdb='python -m debugpy --listen 9339 --wait-for-client'" >> ~/.bashrc

source virtualenv ~/.venv/bin/activate
```

## build
```
make install
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip install -U numpy
```

## train & save model
```
python train.py
cp -r /tmp/1 .
```

## serve
```
sh setup.sh
sh serve.sh
```

## request tensorflow_serving
https://github.com/tensorflow/serving
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md
python request.py
