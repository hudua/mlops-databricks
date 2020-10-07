# Databricks notebook source
# MAGIC %md #### Distributed Deep Learning with Azure Databricks

# COMMAND ----------

# MAGIC %md We will cover some patterns of using Azure Databricks, leveraging distributed computations for deep learning on image classification

# COMMAND ----------

from PIL import Image 
from pyspark.sql.types import StringType, StructType, DoubleType, StructField
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# COMMAND ----------

# MAGIC %md #### Now let's train a number of deep learning models in parallel

# COMMAND ----------

models_df_tune = spark.sql("select * from modelpara2_csv")

# COMMAND ----------

display(models_df_tune)

# COMMAND ----------

# MAGIC %md #### Distributed Deep Learning Model - Hyperparameter tuning
# MAGIC 
# MAGIC ![spark-architecture](https://training.databricks.com/databricks_guide/gentle_introduction/spark_cluster_tasks.png)

# COMMAND ----------

def runDLModelHyper(row):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train[:5000][:][:]
  y_train = y_train[:5000]
  # input image dimensions
  img_rows, img_cols = 28, 28
  # number of classes (digits) to predict
  num_classes = 10

  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(Dropout(0.25))
  model.add(Dropout(row.DropoutA))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  #model.add(Dropout(0.5))
  model.add(Dropout(row.DropoutB))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
  batch_size = 500
  epochs = 1

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  
  score = model.evaluate(x_test, y_test, verbose=0)
  score = float(score[1])
  
  return(row.Model, row.DropoutA, row.DropoutB, score)

# COMMAND ----------

schema = StructType([
  StructField("ModelLocation", StringType(), False),
  StructField("DropoutA", DoubleType(), False),
  StructField("DropoutB", DoubleType(), False),
  StructField("Accuracy", DoubleType(), False)
                    ])

# COMMAND ----------

models_df_tune = models_df_tune.repartition('DropoutA')

# COMMAND ----------

models_df_tune.rdd.getNumPartitions()

# COMMAND ----------

results_df = models_df_tune.rdd.map(runDLModelHyper).toDF(schema)

# COMMAND ----------

# Spark having lazy evaluation, the <display> action actually 'runs' the compute
display(results_df)

# COMMAND ----------

# MAGIC %md #### Distributed Deep Learning Model - Combined with Azure Machine Learning services
# MAGIC 
# MAGIC ![spark-architecture](https://azuremlstudioreadstorage.blob.core.windows.net/edmonton/Capture.JPG)

# COMMAND ----------


