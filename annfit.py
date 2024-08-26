import numpy as np
import tensorflow as tf



def fitNN(X, Y, structure = [25, 25, 25], afunctions = ["sigmoid", "sigmoid", "sigmoid"] ):
    
    #intepolacion lineal para tener valores en cada TPpFP
    #0,5,10
    #X = 0,1,2,3,4,5,6,7,8,9,10
    #sigmoidal expand tails for reinforce learning at those extremes
    stail = 100000
    Xinterp = np.arange(0, X[-1])
    Yinterp = np.interp(Xinterp, X, Y)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-8,
        # "no longer improving" being further defined as "for at least 10 epochs"
        patience=10,
        verbose=1,
        )
    ]

    X_train_e = np.concatenate((np.arange(-stail, np.min(Xinterp)), Xinterp,np.arange(Xinterp[-1]+1, Xinterp[-1]+stail+1)))
    Y_train_e = np.concatenate((np.zeros(stail), Yinterp, np.ones(stail)*Yinterp[-1] ))

    #Normalized TOC
    X_train = X_train_e/X[-1]
    Y_train = Y_train_e/Y[-1]

    X_train_s = X_train[::2]
    Y_train_s = Y_train[::2]

    X_valid_s = X_train[1:][::2]
    Y_valid_s = Y_train[1:][::2]

    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),        # Input layer of size 1
    tf.keras.layers.Dense(structure[0], input_dim = 1, activation= afunctions[0]),                    
    tf.keras.layers.Dense(structure[1], input_dim = structure[0], activation=afunctions[1]),                  
    tf.keras.layers.Dense(structure[2], input_dim = structure[1], activation= afunctions[2]),                   
    tf.keras.layers.Dense(1, input_dim = structure[2], activation='linear')
    ])

        # Compile the model
    model.compile(optimizer = 'adam',  loss = "mse") 

    # Train the model
    model.fit(X_train_s, Y_train_s, validation_data=(X_valid_s, Y_valid_s), epochs=500*2, callbacks=callbacks, verbose=1, batch_size=50)         
    

    yhat = model.predict(X_train)
    realHits = yhat*Y[-1]

    input_data = tf.convert_to_tensor(X_train)
    output = tf.convert_to_tensor(Y_train)
    with tf.GradientTape() as tape2:
        tape2.watch(input_data)
        with tf.GradientTape() as tape1:
            tape1.watch(input_data)
            output = model(input_data)
        first_derivative = tape1.gradient(output, input_data)
    second_derivative = tape2.gradient(first_derivative, input_data)    


    return Xinterp, realHits[stail:-stail+1], first_derivative[stail:-stail+1], second_derivative[stail:-stail+1]