# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./images/center_2017_07_18_21_11_01_154.jpg "DrivingCenter"
[image3]: ./images/left_2017_07_18_21_11_01_154.jpg "DrivingLeft"
[image4]: ./images/right_2017_07_18_21_11_01_154.jpg "DrivingRight"
[image7]: ./images/graph.png "Graph Image"
[image8]: ./images/flip.jpg "Flipped Image"
[image9]: ./images/crop.jpg "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64(model.py lines 100-129).

The model includes RELU layers to introduce nonlinearity (code line 110, 114, 118, 121), and the data is normalized and cropped in the model using a Keras lambda layer (code line 106-107).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 8-22).  
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
I decided not to use dropout layer because, without dropout layer, overfitting seemed not to occur. I judged it from model mean squared error loss graph.


#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 132).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I drove carefully to on the center lane and about two laps.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use LeNet. but simulator result was not good.
o, I decided to use NVIDIA model. but when I run it on the AWS, memory error occurred.
(I used jupyter notebook. that might cause this error which I found after succeeding driving car.)

I couldn't find a true reason, but I thought a number of parameters were the problem. So, I reduced the layer and added a maxpooling layer. Then I could avoid errors.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I augmented data by flipping and using left and right images.

Though augmenting data didn't so change training and validation error loss value, when I checked by simulator vehicle was able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture (model.py lines 100-129) consisted of a convolution neural network with the following layers and layer sizes.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 61, 316, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 61, 316, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 158, 24)   0           activation_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 26, 154, 36)   21636       maxpooling2d_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 26, 154, 36)   0           convolution2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 77, 36)    0           activation_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 9, 73, 48)     43248       maxpooling2d_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 9, 73, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 7, 71, 64)     27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 7, 71, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 31808)         0           activation_4[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           3180900     flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 3,280,891
Trainable params: 3,280,891
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded carefully two laps on track using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I didn't record the vehicle recovering from the left and right sides of the road back to center. I found it was not necessary for this architecture which I judged from "model mean squared error loss" graph.


I put 0.2% of the data into a validation set.  
To augment the training set, I flipped images (model.py line 25-32 ) in half of the probability and used left or right images with a probability of 1/3. When the left or right images were selected, angles were corrected adding or subtracting by 0.2.

Here is the example image of flipping, left and right lane driving:  
![alt text][image8]
![alt text][image3]
![alt text][image4]

After the collection process, I then preprocessed this data by normalizing and cropping(model.py line 107-108).  
Here is the example image of cropping:  

![alt text][image9]


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by graph.  
![alt text][image7]  
```
Epoch 1/5
3346/3346 [==============================] - 147s - loss: 0.0672 - val_loss: 0.0206
Epoch 2/5
3346/3346 [==============================] - 150s - loss: 0.0217 - val_loss: 0.0182
Epoch 3/5
3346/3346 [==============================] - 153s - loss: 0.0212 - val_loss: 0.0227
Epoch 4/5
3346/3346 [==============================] - 148s - loss: 0.0200 - val_loss: 0.0187
Epoch 5/5
3346/3346 [==============================] - 143s - loss: 0.0200 - val_loss: 0.0190
```

I used an Adam optimizer so that manually training the learning rate wasn't necessary.
