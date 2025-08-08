import os
import tensorflow

from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir=os.path.abspath(os.path.join(os.path.dirname(file),'..','dataset'))

train_dir=os.path.join(base_dir,'train')
val_dir=os.path.join(base_dir,'val')
test_dir=os.path.join(base_dir,'test')

train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotational_range=15,
    horizontal_flip=True
)

val_datagen=ImageDataGenerator(rescale=1./255)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator=val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

images,labels=next(train_generator)

print("batch shape :",images.shape,"\nlabels shape :",labels.shape)