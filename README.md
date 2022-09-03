# Traffic-Sign-Recognition
Traffic Sign Detection and Recognition with Deep Learning, CNNs, YOLOv3, and Keras
This research project was conducted under Susan Eileen Fox, DeWitt Wallace Professor and Chair of Mathematics,
Statistics, and Computer Science, at Macalester College

In this project, the classification model was trained using GTSRB data set and the recognition is trained by the GTSDB data set. Only the classification model is included.  

All layers in CNN classification model
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_13 (Conv2D)          (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d_13 (MaxPoolin  (None, 15, 15, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_14 (Conv2D)          (None, 15, 15, 64)        18496     
                                                                 
 max_pooling2d_14 (MaxPoolin  (None, 7, 7, 64)         0         
 g2D)                                                            
                                                                 
 conv2d_15 (Conv2D)          (None, 7, 7, 128)         73856     
                                                                 
 max_pooling2d_15 (MaxPoolin  (None, 3, 3, 128)        0         
 g2D)                                                            
                                                                 
 flatten_4 (Flatten)         (None, 1152)              0         
                                                                 
 dense_8 (Dense)             (None, 128)               147584    
                                                                 
 dense_9 (Dense)             (None, 43)                5547      
                                                                 
=================================================================
Total params: 246,379
Trainable params: 246,379
Non-trainable params: 0
_________________________________________________________________

GTSRB dataset must be downloaded. https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
Before inputting the image through the model, image must me resized to (30, 30)

Example to run the model is below 
#load model
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('/content/drive/MyDrive/GTSRB/signClassify.h5')

# Show the model architecture
new_model.summary()

labels = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120', 'no overtaking', 'no overtaking (trucks)', 
             'priority at next intersection', 'pritority road', 'give way', 'stop', 'no traffic both ways', 'no trucks', 'no entry', 'danger', 'bend left', 'bend right', 'bend', 'uneven road', 'slippery road', 'road narrows',
             'construction', 'traffic signal', 'pedestrian crossing', 'school crossing', 'cycles crossing', 'snow', 'animals', 'restriction ends', 'go right', 'go left', 'go straight', 'go right or straight',
             'go left or straight', 'keep right', 'keep left', 'roundabout', 'restriction ends', 'restriction ends (trucks)']

images = []
image = Image.open('/content/drive/MyDrive/GTSRB/thirtySpeedLimit.jpg')
image = image.resize((30, 30))
imageNp = np.array(image)
#input = np.array([imageNp])
images.append(imageNp)
#new_model.predict(input)

image = Image.open('/content/drive/MyDrive/GTSRB/stopSign.jpeg')
image = image.resize((30, 30))
imageNp = np.array(image)
#input = np.array([imageNp])
images.append(imageNp)

images = np.array(images)
classify = new_model.predict(images)

print("hello", classify)
for i in range(len(classify)):
  index = np.argmax(classify[i])
  print(labels[index])
  
  #Output: 
  hello [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
speed limit 30
stop
