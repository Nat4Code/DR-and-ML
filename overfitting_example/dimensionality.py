import numpy as np
import random
import matplotlib.pyplot as plt

def seven_order_poly(x):
    '''contrive a seven order polynomial to succeed on training data'''
    return 1/20*x*(x+1)*(x-1)*(x-2)*(x+2)*(x-3)*(x+3)

# We contrive data to go in accordance to the aforementioned polynomial:
tr_data = np.array([[-3.038, -1.5], [-3,0], [-2,0], [-1.596,-1.157],
[-0.531,0.618], [0.531,-0.618], [1.596,1.157],[3,0],[3.038,1.5]])

# Extract x & y training data:
x_tr = tr_data[:, 0]; y_tr = tr_data[:, 1]

# Perform a standard linear regression:
m, b = np.polyfit(x_tr, y_tr, 1)

# We generate some test data based on regression: 
x_te = np.zeros(30); y_te = np.zeros(30)
for i in range(30):
    x_val = np.random.uniform(-3.3, 3.3)
    r_y = np.random.uniform(-0.6, 0.6)
    y_val = m*x_val+b+r_y

    x_te[i] = x_val; y_te[i] = y_val

###########################
# Make the training plot: #
#######################################################
# We generate our x & y for our seven-order polynomial:
x = np.linspace(-3.1,3.1,1000); y = seven_order_poly(x)

plt.scatter(x_tr, y_tr, label='Training Data')
plt.plot(x, m*x+b, color='green', label='Linear Order Regression')
plt.plot(x, y, color='red', label='Seventh Order Regression')

# Add training plot details:
plt.title('High Order vs. Linear Regression on Training Set:') 
plt.xlabel('x-data'); plt.ylabel('y-data'); plt.grid(True)
plt.legend(); plt.savefig('training.png'); plt.figure()
##################################################################


##########################
# Make the testing plot: #
##############################################
plt.scatter(x_tr, y_tr, label='Training Data')
plt.scatter(x_te, y_te, color='black', label='Testing Data')
plt.plot(x, m*x+b, color='green', label='Linear Order Regression')
plt.plot(x, y, color='red', label='Seventh Order Regression')

# Add testing plot details:
plt.title('High Order vs. Linear Regression on Testing Set:') 
plt.xlabel('x-data'); plt.ylabel('y-data'); plt.grid(True)
plt.legend(); plt.savefig('testing.png')
##################################################################