## Chaotic Neural Network

### pip install
- numpy
- GPy
- GPyOpt
- Tensorflow
- matplotlib

### How to get the kind of variables in the tensor-graph
~~~python
for op in graph.get_operations():
  print(op.name)
~~~
