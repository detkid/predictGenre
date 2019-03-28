# Result

## Dataset
total words: 103951
低頻度語数: 76377
data words: 27575

## Val data from Twitter with noun
### CNN

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

### RNN
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

## Val data defined by myself with noun
### CNN
```
Using TensorFlow backend.
x_train shape = (106874, 10)
x_val shape = (109, 10)
Build CNN model...
Train on 106874 samples, validate on 109 samples
Epoch 1/4
106874/106874 [==============================] - 44s 415us/step - loss: 2.0513 - acc: 0.3915 - val_loss: 2.0427 - val_acc: 0.4495
Epoch 2/4
106874/106874 [==============================] - 40s 375us/step - loss: 0.7653 - acc: 0.7982 - val_loss: 1.5846 - val_acc: 0.5596
Epoch 3/4
106874/106874 [==============================] - 38s 351us/step - loss: 0.5165 - acc: 0.8556 - val_loss: 1.5686 - val_acc: 0.5413
Epoch 4/4
106874/106874 [==============================] - 37s 347us/step - loss: 0.4171 - acc: 0.8831 - val_loss: 1.6069 - val_acc: 0.5505
```

### RNN
```
Using TensorFlow backend.
x_train shape = (106874, 10)
x_val shape = (109, 10)
Build RNN model...
Train on 106874 samples, validate on 109 samples
Epoch 1/4
106874/106874 [==============================] - 22s 206us/step - loss: 2.3526 - acc: 0.2823 - val_loss: 2.3889 - val_acc: 0.2844
Epoch 2/4
106874/106874 [==============================] - 17s 163us/step - loss: 1.6763 - acc: 0.5956 - val_loss: 2.0839 - val_acc: 0.4404
Epoch 3/4
106874/106874 [==============================] - 18s 166us/step - loss: 1.1203 - acc: 0.7762 - val_loss: 1.9473 - val_acc: 0.4495
Epoch 4/4
106874/106874 [==============================] - 18s 170us/step - loss: 0.8140 - acc: 0.8285 - val_loss: 1.7885 - val_acc: 0.5413
```

## Val data defined by myself with noun, verb
### CNN
test loss: 1.3734066661344755
test accuracy: 0.5504587177836567

### RNN
test loss: 1.8369058665879276
test accuracy: 0.4862385348442498
