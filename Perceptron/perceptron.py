class Perceptron(object):
  def __init__(self, input_num, activator):
    self.activator = activator
    self.weights = [0.0 for _ in range(input_num)]
    self.bias = 0.0

  def __str__(self):
    return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

  def predict(self, input_vec):
    return self.activator(
	reduce(lambda a,b:a+b,
	    map(lambda (x,w):x*w,
		zip(input_vec, self.weights))
	    ,0.0)+self.bias)
  
  def train(self, input_vecs, labels, iterations, lr):
    for i in range(iterations):
	self._one_iteration(input_vecs, labels, lr)

  def _one_iteration(self, input_vecs, labels, lr):
    samples = zip(input_vecs, labels)
    for (input_vec, label) in samples:
	output = self.predict(input_vec)
	self._update_weights(input_vec, output, label, lr)

  def _update_weights(self, input_vec, output, label, lr):
    delta = label - output 
    self.weights = map(
	lambda (x, w): w + lr * delta * x,
	zip(input_vec, self.weights))
    self.bias += lr * delta

 

