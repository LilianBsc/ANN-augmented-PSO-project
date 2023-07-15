import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.array(pd.read_csv("dataset_test_ackley.csv"))

x = data[:, 0:-1]
print(x)
y = data[:, -1]

# load json and create model
modelname = "model_champ2"
json_file = open(modelname + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(modelname + ".h5")
print("Loaded model from disk")
print(type(model))
# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(x, y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# make prediction
ypred = model.predict(x)
classes_y = (ypred > 0.5).astype("int32")

counts = np.bincount(y.astype(int))

print("Rate of positives =", counts[1]/(counts[1] + counts[0]))

matrix = tf.math.confusion_matrix(labels=y, predictions=classes_y).numpy()

print(matrix)

# for i in range(len(classes_y)):
# 	if y[i] == 1:
# 		print("Attempt=%s, Predicted=%s, Correct=%s" % (i, ypred[i], y[i]))