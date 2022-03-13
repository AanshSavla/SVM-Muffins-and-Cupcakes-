# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 20:51:27 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

recipes = pd.read_csv("D:\AanshFolder\datasets\muffins_cupcakes.csv")
#print(recipes.head())

sns.lmplot('Flour','Sugar',data=recipes,hue='Type',palette='Set1',fit_reg=False,scatter_kws={'s':70})

ingredients = recipes[['Flour','Sugar']].values
#print(ingredients)
type_label=np.where(recipes['Type']=='Muffin',0,1)
#print(type_label)
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False,scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black');

sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False,scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],s=80, facecolors='none');

def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('You\'re looking at a muffin recipe!')
    else:
        print('You\'re looking at a cupcake recipe!')
        
print("ENTER QUANTITY OF FLOUR" )
a1 = int (input())
print("ENTER QUANTITY OF SUGAR")
a2 = int (input())
muffin_or_cupcake(a1, a2)
