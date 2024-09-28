import numpy as np
import concurrent.futures
from copy import deepcopy
from tqdm import tqdm
import os
from copy import copy
import sys
import io
import threading
import math
from WAO import WAO
from WAO_DA import WAO_DA

class CECTesting():
  def __init__(self, pop_size=10, generations=100, n_exps=100):
    self.pop_size = pop_size
    self.dims = 10
    self.generations=generations
    self.n_exps = n_exps

  def rastrigin(self, x):
    # Ensure x is a NumPy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    d = len(x)
    return 10 * d + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))+1

  def griewank(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      d = len(x)
      sum1 = np.sum(x**2)
      prod2 = np.prod(np.cos(x / np.sqrt( np.arange(1, d + 1))))
      return sum1 / 4000 - prod2 + 1

  def ackley(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      a = 20
      b = 0.2
      c = 2 * np.pi
      d = len(x)
      sum1 = np.sum(x**2)
      sum2 = np.sum(np.cos(c * x))
      return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)+1

  def happy_cat_function(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      alpha = 1.0 / 8.0
      r2 = np.sum((x - 1.0) ** 2)
      sum_z = np.sum(x - 1.0)

      nx = len(x)
      f = np.power(np.abs(r2 - nx), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5
      return f+1

  def expanded_scaffer6_function(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      y = np.sum(0.5 + (np.sin(np.sqrt(x**2 + np.roll(x, -1)**2))**2 - 0.5) / (1 + 0.001 * (x**2 + np.roll(x, -1)**2))**2)
      return y+1

  def weierstrass_function(self, x, a=0.5, b=3.0, k_max=20):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      n = len(x)
      y = 0.0

      for i in range(n):
          sum_term = 0.0
          sum2_term = 0.0

          for j in range(k_max + 1):
              a_pow_j = a ** j
              b_pow_j = b ** j
              x_term = x[i] + 0.5

              cos_term = math.cos(2.0 * math.pi * b_pow_j * x_term)
              sum_term += a_pow_j * cos_term

              cos2_term = math.cos(2.0 * math.pi * b_pow_j * 0.5)
              sum2_term += a_pow_j * cos2_term

          y += sum_term
      y -= n * sum2_term
      return y+1

  def schwefel_func(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      nx = len(x)
      y = 0.0

      for i in range(nx):
          z = x[i] + 4.209687462275036e+002
          if z > 500:
              y -= (500.0 - (z % 500)) * math.sin(math.sqrt(500.0 - (z % 500)))
              tmp = (z - 500.0) / 100.0
              y += tmp**2 / nx
          elif z < -500:
              y -= (-500.0 + (abs(z) % 500)) * math.sin(math.sqrt(500.0 - (abs(z) % 500)))
              tmp = (z + 500.0) / 100.0
              y += tmp**2 / nx
          else:
              y -= z * math.sin(math.sqrt(abs(z)))

      y += 4.189828872724338e+002 * nx

      return y+1

  def Lennard_Jones(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      D = len(x)
      result = 0
      k = D // 3
      if k < 2:
          k = 2
          D = 6

      sum = 0.0
      for i in range(k - 1):
          for j in range(i + 1, k):
              a = 3 * i
              b = 3 * j
              xd = x[a] - x[b]
              yd = x[a + 1] - x[b + 1]
              zd = x[a + 2] - x[b + 2]
              ed = xd**2 + yd**2 + zd**2
              ud = ed**3
              if ud > 1.0e-10:
                  sum += (1.0 / ud - 2.0) * (1/ud)
              else:
                  sum += 1.0e20

      result += sum
      result += 12.7120622568

      return result+1

  def Hilbert(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      f = 0
      D = len(x)
      b = int(np.sqrt(D))

      sum = 0

      hilbert = np.zeros((b, b))
      y = np.zeros((b, b))

      for i in range(b):
          for j in range(b):
              hilbert[i][j] = 1.0 / np.longdouble(i + j + 1)

      for j in range(b):
          for k in range(b):
              y[j][k] = 0
              for i in range(b):
                  y[j][k] += hilbert[j][i] * x[k + b * i]

      for i in range(b):
          for j in range(b):
              if i == j:
                  sum += np.fabs(y[i][j] - 1)
              else:
                  sum += np.fabs(y[i][j])

      f += sum
      return f+1

  def Chebyshev(self, x):
      if not isinstance(x, np.ndarray):
          x = np.array(x)
      D = len(x)
      f = 0.0
      a = 1.0
      b = 1.2
      px = 0.0
      y = -1.0
      sum_val = 0.0
      dx = 0.0
      dy = 0.0
      sample = 32 * D

      for j in range(D - 2):
          dx = 2.4 * b - a
          a = b
          b = dx

      dy = 2.0 / sample

      for i in range(sample + 1):
          px = x[0]
          for j in range(1, D):
              px = y * px + x[j]
          if px < -1.0 or px > 1.0:
              sum_val += (1.0 - abs(px)) ** 2
          y += dy

      for i in [-1, 1]:
          px = x[0]
          for j in range(1, D):
              px = 1.2 * px + x[j]
          if px < dx:
              sum_val += px ** 2

      f += sum_val
      return f+1

  def wao_manager(self, target_function, boundaries, dims):
    # old_stdout = sys.stdout
    # sys.stdout = mystdout = io.StringIO()
    wao_obj = WAO(fitness_func=target_function,
                  boundaries=boundaries,
                  dimensions=dims,
                  sample_size=self.pop_size,
                  iterations=self.generations)
    wao_solution = wao_obj.wao_runner()
    return wao_solution[0]

  def wao_da_manager(self, target_function, boundaries, dims):
    # old_stdout = sys.stdout
    # sys.stdout = mystdout = io.StringIO()
    wao_obj = WAO_DA(fitness_func=target_function,
                  boundaries=boundaries,
                  dimensions=dims,
                  sample_size=self.pop_size,
                  iterations=self.generations)
    wao_solution = wao_obj.wao_runner()
    return wao_solution[0]

  def generate_report(self):
    target_functions = [
      ("Rastrigin", self.rastrigin, (-100, 100), self.dims),
      ("Griewank", self.griewank, (-100, 100), self.dims),
      ("Ackley", self.ackley, (-100, 100), self.dims),
      ("Happy Cat", self.happy_cat_function, (-100, 100), self.dims),
      ("Expanded Scaffer6", self.expanded_scaffer6_function, (-100, 100), self.dims),
      ("Weierstrass", self.weierstrass_function, (-100, 100), self.dims),
      ("Schwefel", self.schwefel_func, (-100, 100), self.dims),
      ("Lennard_Jones", self.Lennard_Jones, (-4, 4), 18),
      ("Hilbert", self.Hilbert, (-16384, 16384), 16),
      ("Chebyshev", self.Chebyshev, (-8192, 8192), 9)]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for eval_function in tqdm(target_functions, desc="Target Functions"):
        target_function_name, target_function, boundaries, dims = eval_function
        function_results = []
        futures = {executor.submit(self.wao_manager, target_function, boundaries, dims) for _ in tqdm(range(self.n_exps), desc="Experiments")}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            function_results.append(result)
        results.append({"target_function": target_function_name, "D": dims, "search_range": boundaries, "wao_evaluations": function_results})
    return results
