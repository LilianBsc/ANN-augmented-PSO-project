import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def binary_fbeta(ytrue , ypred, beta=2, threshold=0.5, epsilon=1e-7):
    # epsilon is set so as to avoid division by zero error
    
    beta_squared = beta**2 # squaring beta

    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.float32)
    ypred = tf.cast(ypred, tf.float32)

    # setting values of ypred greater than the set threshold to 1 while those lesser to 0
    ypred = tf.cast(tf.greater_equal(ypred, tf.constant(threshold)), tf.float32)

    tp = tf.reduce_sum(ytrue*ypred) # calculating true positives
    predicted_positive = tf.reduce_sum(ypred) # calculating predicted positives
    actual_positive = tf.reduce_sum(ytrue) # calculating actual positives
    
    precision = tp/(predicted_positive+epsilon) # calculating precision
    recall = tp/(actual_positive+epsilon) # calculating recall
    
    # calculating fbeta
    fb = (1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon)

    return fb

data = np.array(pd.read_csv("true_dataset_train_simple.csv"))

x = data[:, 0:-1]
y = data[:, -1]
print(x)
print("input shape =", x.shape[1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
print("input shape =", x_train.shape[1])
counts = np.bincount(y_train.astype(int))

weight_for_0 = 1/counts[0]
weight_for_1 = 1/counts[1]

print("Rate of positives =", counts[1]/(counts[1] + counts[0]))

# data normalisation
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train = (x_train - mean)/std
x_test = (x_test - mean)/std

# buildiing the model
print("input shape =", x_train.shape[1])
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='relu'))

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary() 


# compile the model
metrics = [ tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
            ]

model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), 
              loss='binary_crossentropy',
              metrics=metrics)

# fit the model
print(weight_for_0)
print(weight_for_1)
class_weights = { 0 : weight_for_1,
                  1 : weight_for_0
                }

history = model.fit(x_train, y_train,
                    epochs=50, # you can set this to a big number!
                    batch_size=2048,
                    validation_data = (x_test, y_test),
                    verbose=2,
                    class_weight=class_weights)

# show resultsc
print("fp = ", history.history['val_fp'][-1])
print("fn = ", history.history['val_fn'][-1])

# plt.figure(figsize = (10, 8))
# plt.plot(history.history["val_fp"])
# plt.plot(history.history["val_fn"])
# plt.title("FP and FN rate")
# plt.ylabel('Frequency')
# plt.xlabel("Epoch")
# plt.legend(["False Positive", "False Negative"], loc="upper left")
# plt.show()

# plt.figure(figsize = (10, 8))
# plt.plot(history.history["val_recall"])
# plt.plot(history.history["val_precision"])
# plt.title("Recall (Sensitivity) and Precision (PPV)")
# plt.ylabel('Value')
# plt.xlabel("Epoch")
# plt.legend(["Recall", "Precision"], loc="upper left")
# plt.show()

# plt.figure(figsize = (10, 8))
# plt.plot(history.history["val_accuracy"])
# plt.plot(history.history["val_loss"])
# plt.title("Accuracy and Loss")
# plt.ylabel('Value')
# plt.xlabel("Epoch")
# plt.legend(["Accuracy", "Loss"], loc="upper left")
# plt.show()

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss     :', score[0])
print('Test accuracy :', score[1])

# # plt.plot(history.history['accuracy'])
# # plt.plot(history.history['val_accuracy'])
# # plt.show()
# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.show()

# # model.predict(x_test) 
# # preds = np.round(model.predict(x_test),0) 
# # classes_y = (model.predict(x_test) > 0.5).astype("int32")

# # y_test # 1 and 0 (Heart Disease or not)
# # print(confusion_matrix(y_test, classes_y)) # order matters! (actual, predicted)

# # print(classification_report(y_test, classes_y))

# serialize model to JSON
modelname = "model_champ"
model_json = model.to_json()
with open(modelname + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelname + ".h5")
print("Saved model to disk")

# Print confusion matrix
data = np.array(pd.read_csv("true_dataset_test_simple.csv"))

x = data[:, 0:-1]
y = data[:, -1]

ypred = model.predict(x)
classes_y = (ypred > 0.5).astype("int32")

matrix = tf.math.confusion_matrix(labels=y, predictions=classes_y).numpy()

print(matrix)