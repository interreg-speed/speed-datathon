# SPEED Datathon 2020 CustomVision.ai - Python example

Export your ONNX model, and choose the correct predict.py for your model at this page:

https://github.com/Azure-Samples/customvision-export-samples/tree/main/samples/python/onnx

Make sure your have python 3.5 or higher installed on your computer.

Install ONNX, ONNXRuntime and Pillow with pip:

```
pip install onnx onnxruntime Pillow
```

Then run:

```
python classification_predict.py model.onnx container.jpg
```

If everything is ok, you should now receive the following result:

```
{'classLabel': array([['container']], dtype=object), 'loss': [{'bulk': 0.001739172381348908, 'container': 0.9767897725105286, 'passenger': 0.021471086889505386}]}
```

Yay! Our model is 98% certain the picture of the containership contains, well, an containership! :smile::boat::anchor:

