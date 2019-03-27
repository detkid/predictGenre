# Result

## Dataset
total words: 103951
低頻度語数: 76377
data words: 27575

## CNN

```
Using TensorFlow backend.
x_train shape = (90000, 10)
x_val shape = (16874, 10)
Build CNN model...
Train on 90000 samples, validate on 16874 samples
Epoch 1/4
90000/90000 [==============================] - 34s 375us/step - loss: 2.2005 - acc: 0.3154 - val_loss: 1.4857 - val_acc: 0.6548
Epoch 2/4
90000/90000 [==============================] - 32s 354us/step - loss: 0.9279 - acc: 0.7716 - val_loss: 0.6946 - val_acc: 0.8028
Epoch 3/4
90000/90000 [==============================] - 32s 355us/step - loss: 0.5433 - acc: 0.8501 - val_loss: 0.6380 - val_acc: 0.8161
Epoch 4/4
90000/90000 [==============================] - 32s 356us/step - loss: 0.4297 - acc: 0.8816 - val_loss: 0.6445 - val_acc: 0.8126
```

## RNN
```
Using TensorFlow backend.
x_train shape = (90000, 10)
x_val shape = (16874, 10)
Build RNN model...
Train on 90000 samples, validate on 16874 samples
Epoch 1/4
90000/90000 [==============================] - 19s 214us/step - loss: 2.4000 - acc: 0.1617 - val_loss: 2.2275 - val_acc: 0.2453
Epoch 2/4
90000/90000 [==============================] - 15s 170us/step - loss: 1.8669 - acc: 0.4860 - val_loss: 1.5496 - val_acc: 0.6062
Epoch 3/4
90000/90000 [==============================] - 16s 173us/step - loss: 1.2501 - acc: 0.7278 - val_loss: 1.1292 - val_acc: 0.7601
Epoch 4/4
90000/90000 [==============================] - 15s 169us/step - loss: 0.8787 - acc: 0.8214 - val_loss: 0.9478 - val_acc: 0.7833
```