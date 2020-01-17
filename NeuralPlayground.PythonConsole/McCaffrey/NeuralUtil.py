# helper functions

import numpy as np
import os
os.system('color')
 
def show_vec(v, wid, dec, vals_line):
  print("\033[0m" + "\033[93m", end="") # reset-yellow
  fmt = "% " + str(wid) + "." + str(dec) + "f"  # like % 8.4f
  for i in range(len(v)):
    if i > 0 and i % vals_line == 0: print("")
    print(fmt % v[i] + " ", end="")
  print("")
  print("\033[0m", end="") # reset
  
def show_matrix(m, wid, dec):
  print("\033[0m" + "\033[92m", end="") # reset-green
  fmt = "% " + str(wid) + "." + str(dec) + "f"  # like % 8.4f
  for i in range(len(m)):
    for j in range(len(m[i])):
      x = m[i,j]
      print(fmt % x + " ", end="")
    print("")
  print("\033[0m", end="") # reset

def relu(x):
  if x <= 0.0:
    return 0.0
  else:
    return x

def leaky(x):
  if x <= 0.0:
    return 0.01 * x
  else:
    return x  

def hypertan(x):
  if x < -20.0:
    return -1.0
  elif x > 20.0:
    return 1.0
  else:
    return np.tanh(x)

def log_sig(x):
  if x < -20.0:
    return 0.0
  elif x > 20.0:
    return 1.0
  else:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(vec):
  n = len(vec)
  result = np.zeros(n, dtype=np.float32)
  mx = np.max(vec)
  divisor = 0.0
  for k in range(n):
    divisor += np.exp(vec[k] - mx)
  for k in range(n):
    result[k] =  np.exp(vec[k] - mx) / divisor
  return result

class Erratic:
  def __init__(self, seed):
    self.seed = seed + 0.5

  def next(self):
    x = np.sin(self.seed) * 1000
    result = x - np.floor(x)  # [0.0,1.0)
    self.seed = result  # for next call
    return result

  def next_int(self, lo, hi):
    x = self.next()
    return int(np.trunc((hi - lo) * x + lo))

  def shuffle(self, v):
    n = len(v)
    for i in range(n):
      ri = self.next_int(i, n)
      tmp = v[ri]; v[ri] = v[i]; v[i] = tmp















