import numpy as np
# import tensorflow as tf
# # ran = np.random.randint(1e4,size=1)[0]

# def seed_func():
#     np.random.seed(1)


from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)

# def second_func():
#     # np.random.seed(ran)
#     print(tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32))
    
# seed_func()
# second_func()

from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

print('done')
# fit MLP to dataset and print error
def fit_model(X, y):
    # design network
    model = Sequential()
    model.add(Dense(10, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    model.fit(X, y, epochs=1, batch_size=len(X), verbose=0)
    # forecast
    yhat = model.predict(X, verbose=0)
    print(mean_squared_error(y, yhat[:,0]))

# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to MLP friendly format
values = df.values
X, y = values[:,0], values[:,1]
# repeat experiment
repeats = 10
print('hi')
for _ in range(repeats):
    fit_model(X, y)