# simple artificial neural network using Sigmoid activation function
import numpy as np

class neural_network(object):
	def __init__(self):
		self.inputlayersize = 2
		self.outputlayersize = 1
		self.hiddenlayersize = 3

		#regularization parameter
		Lambda = 0.0001

		#weights
		self.W1 = np.random.randn(self.inputlayersize, self.hiddenlayersize)
		self.W2 = np.random.randn(self.hiddenlayersize, self.outputlayersize)

	# Foward Propagation
	def forward_prop(self, X):
		#propagarte inputs through network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.signmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.signmoid(self.z3)
		return yHat

	# Signmoid activation function
	def signmoid(z):
		return 1/(1+np.exp(-z))

	def sig_deriv():
		return np.exp(-z)/((1+np.exp(-z))**2)

	# def costfunction(self, X, y):
	# 	self.yHat = self.forward_prop(X)
	# 	J = 0.5 * sum((y-self.yHat)**2)

	# def costfunction_deriv(self, X, y):
	# 	self.yHat = self.forward_prop(X)
	# 	delta3 = np.multiply(-(y-sefl.yHat), self.sig_deriv(self.z3))
	# 	dJdW2 = np.dot(self.a2.T, delta3)

	# 	delta2 = np.dot(delta3, self.W2.T)*self.sig_deriv(self.z2)
	# 	dJdW1 = np.dot(X.T, delta2)

	# 	return dJdW1, dJdW2

	#made changes to avoid overfitting
	def costfunction(self, X, y):
	    #Compute cost for given X,y, use weights already stored in class.
	    self.yHat = self.forward(X)
	    #We don't want cost to increase with the number of examples, so normalize by dividing the error term by number of examples(X.shape[0])
	    J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(sum(self.W1**2)+sum(self.W2**2))
	    return J

	def costfunction_deriv(self, X, y):
	    #Compute derivative with respect to W and W2 for a given X and y:
	    self.yHat = self.forward(X)

	    delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
	    #Add gradient of regularization term:
	    dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2

	    delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
	    #Add gradient of regularization term:
	    dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1

	    return dJdW1, dJdW2

    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


    def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

       	return numgrad


