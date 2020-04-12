# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:44:08 2020

@author: Administrator
"""
import random

class gen_ds:
  def random_int_list(self, start, stop, length):
      start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
      length = int(abs(length)) if length else 0
      random_list = []
      for i in range(length):
        random_list.append(random.randint(start, stop))
      return random_list
  
  def random_float_list(self, start, stop, length):
      start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
      length = int(abs(length)) if length else 0
      random_list = []
      for i in range(length):
        random_list.append(random.uniform(start, stop))
      print("random_list1:", random_list)
      return random_list
  
  def gene_z_label(self,seed=None):
    z = self.random_float_list(-1.0, 1.0, 3)
    ls = random.randint(0, 2)
    print("z0:",z)
    return z, ls
  def gen_input_fn(self):
      ds={}
      zs=[]
      lss=[]
      for i in range(8):
          z, ls = self.gene_z_label()
          print("z:", z)
          print("ls:",ls)
          zs.append(z)
          lss.append(ls)
      ds["z"]=zs
      ds["labels"]=lss
      print("zs:",zs)
      print("lss:",lss)
      return ds
g = gen_ds()
g.gen_input_fn()
