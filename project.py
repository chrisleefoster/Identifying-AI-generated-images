import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# function for importing data
def importData():
    # load in training data
    train_data, val_data = tf.keras.utils.image_dataset_from_directory(
        "data/train",               
        image_size = (32,32),       
        color_mode="rgb",           
        label_mode='binary',        
        validation_split = 0.2,     
        subset = "both",                  
        seed = 1                    
    )
    # load in testing data
    test_data = tf.keras.utils.image_dataset_from_directory(
        'data/test/',           
        image_size = (32,32),   
        color_mode="rgb",       
        label_mode='binary'
    )
    return train_data, val_data, test_data

# function for testing different values of alpha, which - as of now - is different learning rates
def testCNN(train_data,val_data,alpha):
    metrics = [0]* len(alpha)
    for a in alpha:
        # reset
        tf.keras.backend.clear_session()

        # build the model
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) 
        model.add(layers.MaxPooling2D((2, 2)))    
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=a),     
                      loss=tf.keras.losses.BinaryCrossentropy(),   
                      metrics=[tf.keras.metrics.F1Score()]
        )                         

        # stop training when the validation loss does not improve
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5,
                                                      restore_best_weights=True
        )  

        # train the model
        trained_model = model.fit(train_data,                 
                            validation_data=val_data,   
                            epochs=100,                 
                            callbacks = [stop_early]
        )
        
        # plot loss and f1-score
        plotTrainingMetrics(trained_model, a)

        # evaluate model
        results = model.evaluate(test_data)
        test_loss = results[0]
        test_f1 = float(results[1])
        metrics[alpha.index(a)] = [a, test_loss, test_f1]
    metrics = np.array(metrics)
    return metrics

def plotTrainingMetrics(model, learning_rate):
    # plot the F1-score of the training and validation set 
    plt.figure()
    plt.plot(model.history['f1_score'], label='f1_score')
    plt.plot(model.history['val_f1_score'], label = 'val_f1_score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig(f'plots/nadam/f1score-lr-{learning_rate}.jpg')
    plt.close()

    # plot the loss of the training and validation set 
    plt.figure()
    plt.plot(model.history['loss'], label='loss')
    plt.plot(model.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.savefig(f'plots/nadam/loss-lr-{learning_rate}.jpg')
    plt.close()

def plotTestMetrics(metrics):
    # plot the f1-score and loss for different learning rates
    fig, ax = plt.subplots(2)
    ax[0].plot(metrics[:,0], metrics[:,2], label='f1_score')
    ax[0].legend(loc='upper right')
    ax[1].plot(metrics[:,0], metrics[:,1], label='loss')
    ax[1].legend(loc='upper right')
    plt.savefig(f'plots/nadam/lr_eval.jpg')
    plt.show()

###################
#   START HERE    #
###################

# get data
train_data, val_data, test_data = importData()

# list of learning rates to test
learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

# train model with various learning rates, returns metrics from testing
test_metrics = testCNN(train_data, val_data, learning_rate)
# returns array with the following values:
# [0,             1,         2      ]
# [learning rate, test_loss, test_f1]

# plot test metrics
plotTestMetrics(test_metrics)