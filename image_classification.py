"""
Created on Wed Jan 11 09:06:28 2023

@author: Wai Yip LIEW (liewwy19@gmail.com)
"""
# %%
#   1. Import packages
import numpy as np
import tensorflow as tf
import os, datetime, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# surpress tensorflow warning messages 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")

#   constant variables
_SEED = 142857
BATCH_SIZE = 64
IMG_SIZE = (224,224)
SOURCE_PATH = os.path.join(os.getcwd(), 'datasets')
SAVED_MODEL_PATH = os.path.join(os.getcwd(),'saved_models')
LOG_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# %%
#   2. Data Loading
#   2.1 load the data into tensorflow dataset using the specific method

train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
                                                   SOURCE_PATH,
                                                   image_size=IMG_SIZE,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   validation_split=0.3,
                                                   subset='both',
                                                   seed=_SEED                                                  
)
class_names = train_dataset.class_names
# %%
#   2.2 display some images from train dataset for quick visual
plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        
# %%
#   3. Data Preprocessing
#   3.1 further split the validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
#   3.2 convert the BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val   = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test  = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
#   3.3 create a small pipeline for data augmentation
data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip('horizontal_and_vertical'))
data_augmentation.add(tf.keras.layers.RandomRotation(0.3))

# %%
#   3.4 apply the data augmentation to test it out
for images,labels in pf_train.take(1):
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(images[0],axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.title(class_names[labels[0]])
        plt.axis('off')

# %%
#   4. Model Development
#   4.1 prepare the layer for data preprocessing
preprocess_input = tf.keras.applications.vgg16.preprocess_input

#   4.2 apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#   4.3 disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
print("Total number of layers:",len(feature_extractor.layers))
tf.keras.utils.plot_model(feature_extractor,show_shapes=True,show_layer_names=True,to_file="feature_extractor_model_summary.png")

# %%
#   4.4 create the classification layers
global_avg = tf.keras.layers.GlobalAveragePooling2D()
output_layer = tf.keras.layers.Dense(len(class_names),activation='softmax')

#   4.5 use functional API to link all of the modules together
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_avg(x)
x = tf.keras.layers.Dropout(0.3)(x) # added to reduce overfitting
outputs = output_layer(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()
# %%
#   4.6 compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#   4.7 evaluate the model before model training
loss0,accuracy0 = model.evaluate(pf_test)
print("Loss (BEFORE Training) =",loss0)
print("Accuracy (BEFORE Training) =",accuracy0)

# %%
#   4.8 define callbacks function
tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)

#Train the model
EPOCHS = 5
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])

# %%
#   5. Apply the fine-tuning learning strategy
feature_extractor.trainable = True

#   5.1 freeze the earlier layers
for layer in feature_extractor.layers[:15]:
    layer.trainable = False

feature_extractor.summary()

#   5.2 compile the updated model
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)      # using a smaller learning_rate to fine tune the model
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#   5.3 continue the model training with this new set of configuration
fine_tune_epoch = 5
total_epoch = EPOCHS + fine_tune_epoch

#   5.4 follow up from the previous model training
history_fine_tune = model.fit(pf_train,validation_data=pf_val,epochs=total_epoch,initial_epoch=history.epoch[-1],callbacks=[tb])

#   5.5 evaluate the final model
test_loss,test_accuracy = model.evaluate(pf_test)
print("Loss (AFTER Training) =", test_loss)
print("Accuracy (AFTER Training) =", test_accuracy)

# %%
#   6. Model Deployment

image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch),axis=1)

#   Compare actual label with prediction
label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))

#   - compute classification report
print(classification_report(label_batch,predictions,target_names=class_names))

#   - confusion matrix visualization
disp = ConfusionMatrixDisplay.from_predictions(label_batch, predictions,cmap='Blues',display_labels=class_names)
ax = disp.ax_.set(title='Confusion Matrix')

#%%
#   7. Model saving
tf.keras.utils.plot_model(model,show_shapes=True,show_layer_names=True,to_file="final_model_summary.png") # saving model summary
model.save(os.path.join(SAVED_MODEL_PATH,'model.h5')) # saving the train model
# %%

