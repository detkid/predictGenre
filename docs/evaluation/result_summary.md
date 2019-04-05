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

              precision    recall  f1-score   support

           0       0.64      0.70      0.67        10
           1       0.89      0.80      0.84        10
           2       0.67      0.40      0.50        10
           3       1.00      0.70      0.82        10
           4       0.50      0.80      0.62        10
           5       1.00      0.30      0.46        10
           6       0.62      0.50      0.56        10
           7       0.60      0.60      0.60        10
           8       0.56      0.50      0.53        10
           9       0.45      0.50      0.48        10
          10       0.35      0.70      0.47        10
         avg       0.59      0.59      0.59       110

### RNN

```
Train on 95595 samples, validate on 10622 samples
Epoch 1/5
95595/95595 [==============================] - 44s 464us/step - loss: 2.2013 - acc: 0.2800 - val_loss: 1.7289 - val_acc: 0.5149
Epoch 2/5
95595/95595 [==============================] - 40s 415us/step - loss: 1.2603 - acc: 0.6324 - val_loss: 0.8970 - val_acc: 0.7661
Epoch 3/5
95595/95595 [==============================] - 38s 402us/step - loss: 0.6252 - acc: 0.8392 - val_loss: 0.6598 - val_acc: 0.8280
Epoch 4/5
95595/95595 [==============================] - 38s 395us/step - loss: 0.4115 - acc: 0.8944 - val_loss: 0.6270 - val_acc: 0.8354
Epoch 5/5
95595/95595 [==============================] - 44s 465us/step - loss: 0.3125 - acc: 0.9192 - val_loss: 0.6340 - val_acc: 0.8371
```

test loss: 1.4616667790846392  
test accuracy: 0.5727272716435519

              precision    recall  f1-score   support

           0       0.33      1.00      0.50        10
           1       0.64      0.70      0.67        10
           2       0.57      0.40      0.47        10
           3       0.86      0.60      0.71        10
           4       0.62      0.80      0.70        10
           5       0.75      0.30      0.43        10
           6       1.00      0.40      0.57        10
           7       0.78      0.70      0.74        10
           8       0.75      0.30      0.43        10
           9       0.50      0.40      0.44        10
          10       0.54      0.70      0.61        10
         avg       0.57      0.57      0.57       110

## characterized learning data

total words: 5173  
低頻度語数: 1463  
total words: 3711  

## CNN
test loss: 1.1790448719804938  
test accuracy: 0.663636361468922

              precision    recall  f1-score   support

           0       0.62      0.80      0.70        10
           1       0.50      0.90      0.64        10
           2       1.00      0.70      0.82        10
           3       0.75      0.90      0.82        10
           4       0.64      0.90      0.75        10
           5       0.43      0.30      0.35        10
           6       0.80      0.40      0.53        10
           7       1.00      0.60      0.75        10
           8       0.67      0.20      0.31        10
           9       0.47      0.70      0.56        10
          10       0.90      0.90      0.90        10
          avg      0.66      0.66      0.66       110
   
## RNN
test loss: 1.4250908981670032  
test accuracy: 0.5909090941602534

              precision    recall  f1-score   support

           0       0.62      0.50      0.56        10
           1       0.50      0.90      0.64        10
           2       0.73      0.80      0.76        10
           3       0.53      0.90      0.67        10
           4       0.40      0.60      0.48        10
           5       1.00      0.20      0.33        10
           6       1.00      0.40      0.57        10
           7       0.89      0.80      0.84        10
           8       0.33      0.10      0.15        10
           9       0.40      0.60      0.48        10
          10       0.88      0.70      0.78        10
         avg       0.59      0.59      0.59       110
