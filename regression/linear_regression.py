from __future__ import division
from itertools import izip
import numpy as np

class LinearRegression:
    """
    Linear regression with batch gradient descent 
    """
    
    def __init__(self, learning_rate = 0.001, max_iterations = 1000):
        self.alpha    = learning_rate
        self.max_iter = max_iterations
        self.convergance_factor = 1e-07

    def fit(self, X_train, Y_train, sgd = False):
        """
        Fit a linear regression model on the training set 
        using batch gradient descent and store the model 
        
        @param X_train  : training set X 
        @param Y_traing : training set Y 
        @param SGD      : Apply stochastic gradient descent 
        """
        if type(X_train) == list:
            X_train = np.asarray(X_train)
        if type(Y_train) == list:
            Y_train = np.asarray(Y_train)
        
        self.X_train,self.Y_train = X_train,Y_train
        self.num_samples,self.num_features = self.X_train.shape
        self.num_outputs, = self.Y_train.shape
        
        if self.num_outputs != self.num_samples:
            raise MalformedModel("Y_train must be of shape of (X_train[0], 1)")
        
        #initilize the model params with uniform distribution in [0,1]
        self.model = np.reshape(np.random.rand(self.num_features + 1), (self.num_features + 1, 1))
        # set the bais model weight to 1
        self.model[self.num_features][0] = 1
        
        train_algo = self.__batch_train if not sgd else self.__stochastic_train
        
        self.iterations = 0
        while not self.__convergence():
            train_algo()
            
    
    def __batch_train(self):
        """
        batch mode of training, batch gradient descent
        """
        self.old_model  = np.copy(self.model)
        for index, theta in enumerate(self.old_model):
                
            gradient = 0
            for train_example, target in izip(self.X_train, self.Y_train):
                model_at_example = np.dot(train_example, self.old_model[:-1]) + self.old_model[self.num_features]
                #non bias input 
                if index < self.num_features: 
                    gradient += ((target - model_at_example) * train_example[index])
                else:
                    gradient += (target - model_at_example)
                
            theta = theta + gradient * self.alpha
            self.model[index][0] = theta
        print self.model
            
    def __stochastic_train(self, learning_rate_delta = False):
        """
        stochastic model of training, a.k.a SGD
        @param learning_rate_delta : Whether to vary the learning rate with iterations 
        """
        for train_example, target in izip(self.X_train, self.Y_train):
            
            self.old_model  = np.copy(self.model)
            model_at_example = np.dot(train_example, self.old_model[:-1]) + self.old_model[self.num_features]
            for index, theta in enumerate(self.old_model):
                #non bias input 
                if index < self.num_features: 
                    gradient = ((target - model_at_example) * train_example[index])
                else:
                    gradient = (target - model_at_example)
                
                theta = theta + gradient * self.alpha
                self.model[index][0] = theta
            print self.model

    def predict(self, test_X):
        """
        predict the Y for the X_test based on the computed model
        
        @param  test_X : test set X
        @return predicted Y value for each x in X as list
        """
        if type(test_X) == list: test_X = np.asarray(test_X)
        num_samples, num_features = test_X.shape
        
        test_Y = np.ndarray(shape = (num_samples,))
        if num_features + 1 != self.model.shape[0]: 
            raise MalformedModel("test set feature space size does not match model")
        
        transposed_model = np.transpose(self.model[:-1])
        for index, test_input in enumerate(test_X):
            predicted = np.dot(transposed_model, test_input) + self.model[-1:]
            test_Y[index] = predicted
        
        return test_Y

    def __convergence(self):
        """
        Check convergance of the model
        """
        try:
            self.old_model
        except AttributeError, e:
            return False
        
        theta_converged = True
        self.iterations += 1
        
        for index, old_theta in enumerate(self.old_model):
            if np.abs(old_theta - self.model[index][0]) > self.convergance_factor:
                theta_converged = False
                break
        
        if self.iterations >= self.max_iter: return True
        if theta_converged: return True
        else: return False
