import copy
import numpy as np
import random
import pickle

configuration = 'toughness'
if configuration == 'toughness':
  NMATERIAL = 2
else:
  NMATERIAL = 3
HASHSET_SIZE = 100000007

class Particle:
  def __init__(self, res):
    self.res = res
    self.configuration = np.zeros(shape=res, dtype=np.int8)
    self.coord = None
    self.curve = []
    self.extra = {}
    for i in range(res[0]):
      for j in range(res[1]):
        self.configuration[i, j] = random.randrange(NMATERIAL)
    self.mirror()

  def mirror(self):
    if configuration == 'toughness':
      for i in range(self.res[0]):
        for j in range(self.res[1]):
          x = min(i, self.res[0] - 1 - i)
          y = min(j, self.res[1] - 1 - j)
          self.configuration[i, j] = self.configuration[x, y]
    else:
      for i in range(self.res[0]):
        for j in range(self.res[1]):
          x = min(i, self.res[0] - i)
          y = min(j, self.res[1] - j)
          if x > y:
            x, y = y, x
          self.configuration[i, j] = self.configuration[x, y]

  def mutate(self):
    p = random.randrange(10000)
    if p < 100:
      # 1% flip
      self.configuration = NMATERIAL - 1 - self.configuration
    elif p < 1100:
      # 10% draw line
      start_x = random.randrange(0, self.res[0] // 2)
      start_y = random.randrange(0, self.res[1] // 2)
      end_x = random.randrange(0, self.res[0] // 2)
      end_y = random.randrange(0, self.res[1] // 2)
      steps = self.res[0] * 2
      new_mat = random.randrange(NMATERIAL)
      for i in range(steps):
        x = int((end_x - start_x) * (i + 0.5) / steps + 0.5 + start_x)
        y = int((end_y - start_y) * (i + 0.5) / steps + 0.5 + start_y)
        self.configuration[x][y] = new_mat

    else:
      # 89% jitter voxel
      i = random.randrange(self.res[0] // 2)
      j = random.randrange(self.res[1] // 2)
      original = self.configuration[i, j]
      while True:
        new_mat = random.randrange(NMATERIAL)
        if new_mat != original:
          break
      if p < 9700:
        # 86% small change
        self.configuration[i, j] = new_mat
      else:
        # 3 % big change
        size = random.randrange(self.res[0] // 2)
        for x in range(size):
          for y in range(size):
            self.configuration[min(i + x, self.res[0]), min(j + y, self.res[1])] = new_mat

    self.mirror()

  def hash(self):
    ret = 0
    for i in range(self.res[0]):
      for j in range(self.res[1]):
        ret = (ret * 99997 + self.configuration[i][j]) % HASHSET_SIZE
    return ret
    