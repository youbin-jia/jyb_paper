# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

test_txt = 'C:\\Users\\Administrator\\Desktop\\global_d_loss_filted.txt'
def load_label_set(label_dir):
 label_folder = open(label_dir, "r")
 loss=[]
 acc=[]
 x=[]
 trainlines = label_folder.read().splitlines() #返回每一行的数据
 i=1
 for line in trainlines:
     #line = line.split(" ") #按照空格键分割每一行里面的数据
     line = re.split(" |\,", line)
     print(line)
     if len(line)<5:
         continue
     i+=1
     x.append(i)
     loss.append(float(line[6]))
     print(i," ",float(line[6]))
     #acc.append(float(line[10]))#box读取标签ground_truth
     label_folder.close()

 return x,loss,acc
x,loss,acc = load_label_set(test_txt)

plt.figure(figsize=(10,6))
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.plot(x,loss,color='r', linestyle='-')
plt.xlabel(u"训练轮次 （Epoch）",fontsize=12)#fill the meaning of X axis
plt.ylabel(u"分类误差  (Loss-class)",fontsize=12)#fill the meaning of Y axis
#plt.title(u'sin(x)')#add the title of the figure
#在ipython的交互环境中需要这句话才能显示出来
plt.legend()#显示左下角的图例
plt.show()

