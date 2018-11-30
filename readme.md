# Chaotic Neural Network

## pip install
- numpy
- GPy
- GPyOpt
- tensorflow
- tensorflow_probability
- matplotlib
- pyaudio (you need to install portaudio19-dev(apt-get))

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


## How to optimize the hyper-parameters
We can optimize hyper parameters with GPy and GPyOpt libraries.
Their libraries can easily execute Bayesian-Optimization.

~~~python
	import GPy
	import GPyOpt

	bounds = [{'name': 'kf',    'type': 'continuous',  'domain': (0.0, 100.0)},
		{'name': 'kr',    'type': 'continuous',  'domain': (0.0, 100.0)},
		{'name': 'alpha', 'type': 'continuous',  'domain': (0.0, 100.0)}]

	# Advenced search
	opt_network = GPyOpt.methods.BayesianOptimization(f=opt, domain=bounds)

	# Search optimal parameters
	opt_network.run_optimization(max_iter=10)
	print("optimized parameters: {0}".format(opt_network.x_opt))
	print("optimized loss: {0}".format(opt_network.fx_opt))
~~~

'f' is the function which you wanna minimize.
You can optimize the arguments of the function.
In detail, see my code.

As for caution, the parameters which you wanna optimize must be written by the order(continuous-params, discrete-params).


## How to control with Joystick-Controller
'Qjoypad' package is efficient to use easily.
The following commands use to treat the package.

~~~
$ sudo apt-get install qjoypad
$ qjoypad -notray
~~~






