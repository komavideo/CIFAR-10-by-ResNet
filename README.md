ResNet残差网络再挑战 - CIFAR-10
=============================

## 知识点

* ResNet简介
* 利用ResNet残差网络再挑战CIFAR-10

## 官网

https://www.microsoft.com/en-us/research/

## ResNet简介

ResNet是在2015年由微软研究院的院士 Kaiming He 提出的神经网络模型。
在ResNet里面有一种叫做“残差块”的短构造，它可以跳过一些不必要学习的计算，
解决了普通CNN模型的无法处理深层模型中的各种性能恶化的问题。

### 知乎

https://zhuanlan.zhihu.com/p/42706477

## ResNet论文

### Deep Residual Learning for Image Recognition

https://arxiv.org/abs/1512.03385

### Identity Mappings in Deep Residual Networks

https://arxiv.org/abs/1603.05027

## 实战演习

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

os.system("cls")

#############################################
# 生成卷积层
def conv(filters, kernel_size, strides=1):
    return Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))

#############################################
# 残差块A
def first_residual_unit(filters, strides):
    def f(x):
        # →BN→ReLU
        x = BatchNormalization()(x)
        b = Activation('relu')(x)

        # 卷积层→BN→ReLU
        x = conv(filters // 4, 1, strides)(b)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 卷积层→BN→ReLU
        x = conv(filters // 4, 3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 卷积层→
        x = conv(filters, 1)(x)

        # 形状调整
        sc = conv(filters, 1, strides)(b)

        return Add()([x, sc])
    return f

#############################################
# 残差块B
def residual_unit(filters):
    def f(x):
        sc = x
        
        # →BN→ReLU
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 卷积层→BN→ReLU
        x = conv(filters // 4, 1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 卷积层→BN→ReLU
        x = conv(filters // 4, 3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 卷积层→
        x = conv(filters, 1)(x)

        return Add()([x, sc])
    return f

#############################################
# 残差块A和残差块B*17生成
def residual_block(filters, strides, unit_size):
    def f(x):
        x = first_residual_unit(filters, strides)(x)
        for i in range(unit_size-1):
            x = residual_unit(filters)(x)
        return x
    return f

#############################################
# LearningRateScheduler
def step_decay(epoch):
    x = 0.1
    if epoch >= 80: x = 0.01
    if epoch >= 120: x = 0.001
    return x

def run():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    print("train_images.shape:", train_images.shape)
    print("train_labels.shape:", train_labels.shape)
    print("test_images.shape:", test_images.shape)
    print("test_labels.shape:", test_labels.shape)

    # 数据集预处理
    train_images = train_images.astype('float32')/255.0
    test_images = test_images.astype('float32')/255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # 输入形状
    input = Input(shape=(32,32, 3))

    # 卷积层
    x = conv(16, 3)(input)

    # 残差块 x 54
    x = residual_block(64, 1, 18)(x)
    x = residual_block(128, 2, 18)(x)
    x = residual_block(256, 2, 18)(x)

    # →BN→ReLU
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 池层
    x = GlobalAveragePooling2D()(x)

    # 全连接层
    output = Dense(10, activation='softmax', kernel_regularizer=l2(0.0001))(x)

    # 生成模型
    model_filename = "./resnet.h5"
    model = Model(inputs=input, outputs=output)

    # 模型编译
    model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['acc'])
    # model.summary()

    # 准备图片：ImageDataGenerator
    train_gen  = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True,
        width_shift_range=0.125, 
        height_shift_range=0.125, 
        horizontal_flip=True)
    test_gen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True)

    # 数据集前计算
    for data in (train_gen, test_gen):
        data.fit(train_images)

    lr_decay = LearningRateScheduler(step_decay)

    # 开始训练
    batch_size = 128
    history = model.fit(
        train_gen.flow(train_images, train_labels, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=train_images.shape[0] // batch_size,
        validation_data=test_gen.flow(test_images, test_labels, batch_size=batch_size),
        validation_steps=test_images.shape[0] // batch_size,
        callbacks=[lr_decay])

    # 模型保存
    model.save(model_filename)

    # 显示训练结果
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

    # 结果评价
    batch_size = 128
    test_loss, test_acc = model.evaluate_generator(
        test_gen.flow(test_images, test_labels, batch_size=batch_size),
        steps=10)
    print('val_loss: {:.3f}\nval_acc: {:.3f}'.format(test_loss, test_acc ))

    # 结果图形显示
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[i])
    plt.show()

    test_predictions = model.predict_generator(
        test_gen.flow(test_images[0:10], shuffle = False, batch_size=1),
        steps=10)
    test_predictions = np.argmax(test_predictions, axis=1)
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']
    print([labels[n] for n in test_predictions])
    print([n for n in test_predictions])

run()
```

## 课程文件

https://github.com/komavideo/CIFAR-10-by-ResNet

## 小马视频频道

http://komavideo.com
