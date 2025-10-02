#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center; color:green";> ISAT 449: Emerging Topics in Applied Data Science </h1>
# 
# <img src="mnist.png" alt="description" width="400" style="display:block; margin-left:auto; margin-right:auto;">
# 
# <h2 style="color:blue; font-family:georgia; text-align:center;">
#   Lab3 -Training a Multilayer Perceptron Model using Scikit-Learn
# (Image Recognition of the MNIST Handwritten Digits Dataset)
# </h2>
# 
# <h3 style="color:blue; font-family:georgia; text-align:center;">
#   (Image Recognition of the Scikit Learn Handwritten Digits Dataset)
# </h3>
# 
# **By: Jaylon Taylor**

# <h3 style="text-align:left; color:black";> MNIST classification with SciKit-Learn using MLPClassifier
# </h3>
# 
# This notebook shows how to define and train a simple Neural-Network with SciKit-Learn's MLPClasifier.

# <h3 style="text-align:left; color:black";> Learning Objectives
# </h3>
# - Import and learn about the MNIST dataset and its structure
# - Create training, validation and test datasets from the MNIST data
# - Build a simple but effective Scikit Learn MLPClassifier model for recognizing digits, based on looking at every pixel in the image.

# In[36]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
# Load data from https://www.openml.org/d/554
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
mnist = fetch_openml(name='mnist_784',parser='pandas',version=1, cache=False)
mnist.data.shape


# When one learns how to program, there's a tradition that the first thing you do is print "Hello World." Just like programming has Hello World,
# machine learning has MNIST.
# MNIST is a simple computer vision dataset. It consists of images of handwritten digits like these:

# <img src="mnist_image.png" alt="description" width="400" style="display:block; margin-left:auto; margin-right:auto;">

# <h3 style="text-align:left; color:black";> Loading Data
# </h3>
# 
# Using SciKit-Learns fetch_openml to load MNIST data.

# <h3 style="text-align:left; color:black";> Print the Shape of the data
# </h3>

# In[37]:


mnist.data.shape


# <h3 style="text-align:left; color:black";> Preprocessing Data
# </h3>

# In[38]:


X = mnist.data.astype('float32')
y = mnist.target.astype('int64')


# In[39]:


X /= 255.0


# In[40]:


X.min(), X.max()


# In[41]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[42]:


X_train.shape, y_train.shape


# In[43]:


def plot_example(X, y):
    """Plot the first 5 images and their labels in a row."""
    for i, (img, y) in enumerate(zip(X[:5].values.reshape(5, 28, 28), y[:5])):
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(y)


# In[44]:


plot_example(X_train, y_train)


# In[45]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
ax.hist(y,bins=[0,1,2,3,4,5,6,7,8,9,10], edgecolor="b", histtype="bar",align='left')
ax.set_title('Histogram: Training data set')
ax.set(xlabel='Number', ylabel='Frequency')
ax.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9] );
ax.axhline(y=(y.size/10), label="average frequency",linestyle='dashed', color='r')
ax.legend()


# In[46]:


from sklearn.neural_network import MLPClassifier

#TO-DO Instantiate the model and name it as below
mlp_mnist_model = MLPClassifier(
solver='adam',
hidden_layer_sizes=(256, 128, 32),
learning_rate_init= 0.001,
random_state=1,
max_iter=500,
verbose=True,
tol=1.0e-2)


# In[47]:


mlp_mnist_model.fit(X_train, y_train)


# In[48]:


mlp_mnist_model.out_activation_


# In[49]:


mlp_mnist_model.n_layers_


# In[50]:


mlp_mnist_model.n_outputs_


# In[51]:


mlp_mnist_model.n_iter_


# <h3 style="text-align:left; color:black";> Model Evaluation and Peformance
# </h3>

# In[52]:


mlp_mnist_model.score(X_test, y_test)


# In[53]:


from sklearn.metrics import accuracy_score


# In[54]:


y_pred = mlp_mnist_model.predict(X_test)


# In[55]:


accuracy_score(y_test, y_pred)


# In[56]:


error_mask = y_pred != y_test


# In[57]:


plot_example(X_test[error_mask], y_pred[error_mask])


# In[58]:


loss_values = mlp_mnist_model.loss_curve_


# In[59]:


print('loss values are {}'.format(loss_values))


# In[60]:


# TO-DO Plot the loss_values (Label each axis, put in a legend and put a title on plot.)


# In[61]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 4))
plt.plot(loss_values, color="blue", label = 'training loss')
plt.xlabel('epoch')
plt.ylabel('loss');
#legend
plt.legend(loc='upper right')
#title
plt.title('Training Loss Per Epoch: MNIST Digits Dataset ')
#show plot
plt.show()


# <h3 style="text-align:left; color:black";> Classification Report
# </h3>

# In[62]:


from sklearn import metrics
print ("Classification Report:")
print (metrics.classification_report(y_test, y_pred))
#TO-DO: print a pretty confusion matrix (see previous notebooks)


# <h3 style="text-align:left; color:black";> Confusion Matrix
# </h3>

# In[63]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Print confusion matrix
y_pred = np.round(mlp_mnist_model.predict(X_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[])
disp.plot()
plt.show()


# <h3 style="text-align:left; color:black";>Exercise 1 (5 points)
# </h3>
# 
# Based on the confusion matrix, which digit was hardest for the model to classify? Justify your answer!.
# 
# **The hardest digit to classify was 5 due to having the lowest recall and missclassifications out of all the numbers. It was wrong 24 times out of 892 times. Having the lowest support also indicates it was the least stable**

# In[64]:


##The hardest digit to classify was #5 based on the confusion matrix


# <h3 style="text-align:left; color:black";>Exercise 2 (5 points)
# </h3>
# 
# Based on the output of your classification report, out of all the times digit 7 should have been predicted, what percentage of times
# was it correctly predicted? (HINT: You may need to research the meaning and difference between precision, recall and f1-score)
# Do a simple calculation (do it here!) to justify your answer. It compares to what metric in the classification report?
# 
# **Number 7 was correctly predicted 1001 times of 1028 times from the classification report. The precision was 98% meaning the amount of positive predictions were correct. Recall was 97% meaning the amount of actual positives I was able to find. F1-score is the balance score between precision and recall.**
# 
# **Calculation for recall would be 1001/1028 = .9737 or 97.4%.
# Calculation for precision would be 20 False Positives, so 981/1001 = .98 or 98%
# Calculation for f1-score would be 2(0.98)(.974)/(0.98)+(0.97) = .977 or around 98%.**
# 

# <h3 style="text-align:left; color:black";>Exercise 3 (10 points)
# </h3>
# 
# Train the model using the 'sgd' optimizer (scikit learn calls it a solver). Does the accuracy improve? Comment (one line!) as to what you
# think is contributing to the change in performance of the model.
# 
# **The model doesn't improve as sgd is less accurate than the adam optimizer due to being more sensitive than the adam optimzer and converges at a slower rate than adam as well.**

# In[65]:


sgd_mlp_digits_model = MLPClassifier(
    solver='sgd',
    hidden_layer_sizes=(256, 128, 32),
    learning_rate_init=0.001,
    random_state=1,
    max_iter=500,
    verbose=True,
    tol=1.0e-2
)
sgd_mlp_digits_model.fit(X_train, y_train)
print(f'The accuracy of the model is: {100*sgd_mlp_digits_model.score(X_test, y_test):0.2f}%')


# <h3 style="text-align:left; color:black";>Exercise 4 (10 points)
# </h3>
# 
# Feed the model a single image from the test set to classify and have the model return the class membership probabilities and print them
# out for this single image classification. Note you have to reshape the imgage to 28 x 28 before you try and display it! Also, you can use the
# last model you trained if your accuracy is > 95%
# NOTE:For this exercise, feed the model the image: X_test[54:55]

# In[78]:


import numpy as np

single_image = X_test.iloc[54:55]  #row 54 

single_image_np = np.array(single_image).reshape(28, 28)

probabilities = mlp_mnist_model.predict_proba(single_image)

print("Class membership probabilities are:\n", probabilities)
print("Sum of the probabilities is:", np.sum(probabilities))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(single_image_np, cmap="gray")
plt.title(f"Predicted Digit: {np.argmax(probabilities)}")
plt.axis("off")
plt.show()


# In[80]:


actual_label = y_test.iloc[54]

print(f"Actual Digit: {actual_label}")


# In[ ]:




