# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:30:42 2020

@author: Administrator
"""
import os

file=open("C:\\Users\\Administrator\\Desktop\\2020-4-20-d_loss.txt")
wf=open('C:\\Users\\Administrator\\Desktop\\2020-4-20-d_loss_filted.txt','w')
line=file.readline() 
while line:
    if line.find('loss = ')>0:
        wf.writelines(line)
    line=file.readline() 
