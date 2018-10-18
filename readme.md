# Chaotic Neural Network

## pip install
- numpy
- GPy
- GPyOpt
- Tensorflow
- matplotlib

## How to save or restore your session in tensorflow
### How to save
#### save all parameters
~~~python
  saver = tf.train.Saver()
  saver.save(sess, model_path + 'model.ckpt')
~~~

#### save specific parameters
~~~python
  # You need to write to want the parameters-list in the argument of Saver()
  saver = tf.train.Saver([Wi, Wo])
  saver.save(sess, model_path + 'model.ckpt')
~~~

### How to restore
~~~python
  saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta')
  saver.restore(psess, tf.train.latest_checkpoint(model_path))

  graph = tf.get_default_graph()

  # You have to confirm the registered variable name
  for op in graph.get_operations():
    print(op.name)

  Wi = graph.get_tensor_by_name("Wi/Wi:0")
  Wo = graph.get_tensor_by_name("Wo/Wo:0")
~~~

