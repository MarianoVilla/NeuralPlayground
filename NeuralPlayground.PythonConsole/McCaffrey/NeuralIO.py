# neural_io.py
# Python 3.6.5
#Source: https://visualstudiomagazine.com/Articles/2019/07/01/modern-neural-networks.aspx?Page=1

import NeuralUtil as U

import numpy as np

# ------------------------------------------------------------------------------

class NeuralNet:

  def __init__(self, num_input, num_hidden, num_output):
    self.ni = num_input
    self.nh = num_hidden
    self.no = num_output
	
    self.i_nodes = np.zeros(shape=[self.ni], dtype=np.float32)
    self.h_nodes = np.zeros(shape=[self.nh], dtype=np.float32)
    self.o_nodes = np.zeros(shape=[self.no], dtype=np.float32)
	
    self.ih_weights = np.zeros(shape=[self.ni,self.nh], dtype=np.float32)
    self.ho_weights = np.zeros(shape=[self.nh,self.no], dtype=np.float32)
	
    self.h_biases = np.zeros(shape=[self.nh], dtype=np.float32)
    self.o_biases = np.zeros(shape=[self.no], dtype=np.float32)

    self.h_sums = np.zeros(shape=[self.nh], dtype=np.float32)
    self.o_sums = np.zeros(shape=[self.no], dtype=np.float32)
	
  def set_weights(self, weights):
    idx = 0
    for i in range(self.ni):
      for j in range(self.nh):
        self.ih_weights[i,j] = weights[idx]
        idx += 1
		
    for j in range(self.nh):
      self.h_biases[j] = weights[idx]
      idx += 1

    for j in range(self.nh):
      for k in range(self.no):
        self.ho_weights[j,k] = weights[idx]
        idx += 1
	  
    for k in range(self.no):
      self.o_biases[k] = weights[idx]
      idx += 1
	  
  def get_weights(self):
    tw = (self.ni * self.nh) + self.nh + (self.nh * self.no) + self.no
    result = np.zeros(shape=[tw], dtype=np.float32)
    idx = 0  # points into result
    
    for i in range(self.ni):
      for j in range(self.nh):
        result[idx] = self.ih_weights[i,j]
        idx += 1
		
    for j in range(self.nh):
      result[idx] = self.h_biases[j]
      idx += 1

    for j in range(self.nh):
      for k in range(self.no):
        result[idx] = self.ho_weights[j,k]
        idx += 1
	  
    for k in range(self.no):
      result[idx] = self.o_biases[k]
      idx += 1
	  
    return result
 	
  def eval(self, x_values):
    self.i_nodes = x_values  # by ref

    for j in range(self.nh):
      for i in range(self.ni):
        self.h_sums[j] += self.i_nodes[i] * self.ih_weights[i,j]
      self.h_sums[j] += self.h_biases[j]  # add the bias

    print("\nPre-activation hidden node values: ")
    U.show_vec(self.h_sums, 8, 4, len(self.h_sums))

    for j in range(self.nh):    
      self.h_nodes[j] = U.leaky(self.h_sums[j])  # activation

    print("\nPost-activation hidden node values: ")
    U.show_vec(self.h_nodes, 8, 4, len(self.h_sums))

    for k in range(self.no):
      for j in range(self.nh):
        self.o_sums[k] += self.h_nodes[j] * self.ho_weights[j,k]
      self.o_sums[k] += self.o_biases[k]
    print("\nPre-activation output node values: ")
    U.show_vec(self.o_sums, 8, 4, len(self.o_sums))
 
    softout = U.softmax(self.o_sums)
    for k in range(self.no):
      self.o_nodes[k] = softout[k]
	  
    result = np.zeros(shape=self.no, dtype=np.float32)
    for k in range(self.no):
      result[k] = self.o_nodes[k]
	  
    return result

# end class NeuralNet

# ------------------------------------------------------------------------------

def main():
  print("\nBegin NN with leaky ReLU and softmax IO demo ")

  # 1. create network
  print("\nCreating a 3-4-2 leaky ReLU, softmax NN")
  nn = NeuralNet(3, 4, 2) 

  # 2. set weights and biases
  wts = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
    0.07, 0.08, 0.09, 0.10, 0.11, 0.12,  # ih weights
    0.13, 0.14, 0.15, 0.16,  # h biases
    0.17, 0.18, 0.19, 0.20,  # ho weights    
    0.21, 0.22, 0.23, 0.24, 
    0.25, 0.26], dtype=np.float32)  # o biases

  print("\nSetting weights and biases")
  U.show_vec(wts, wid=6, dec=2, vals_line=8)
  nn.set_weights(wts)

  # 3. set input
  X = np.array([3.0, 4.0, -4.5], dtype=np.float32)
  print("\nSetting inputs to: ")
  U.show_vec(X, 6, 2, len(X))

  # 4. compute outputs
  print("\nComputing output values . . ")
  Y = nn.eval(X)
  print("\nOutput values: ")
  U.show_vec(Y, 8, 4, len(Y))

  print("\nEnd IO demo ")
   
if __name__ == "__main__":
  main()
