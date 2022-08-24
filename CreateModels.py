import os
import tensorflow as tf


def makemodel(file_path, iteration):
    try:
        iteration = int(iteration)
        batch_size = 32
        img_height = 224
        img_width = 224
        TRAIN_DIR = os.path.join(file_path, "train")
        VALID_DIR = os.path.join(file_path, "validation")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(TRAIN_DIR,
                                                                       seed=2509,
                                                                       image_size=(img_height, img_width),
                                                                       batch_size=batch_size)

        valid_ds = tf.keras.preprocessing.image_dataset_from_directory(VALID_DIR,
                                                                       seed=2509,
                                                                       image_size=(img_height, img_width),
                                                                       shuffle=False,
                                                                       batch_size=batch_size)
        class_names = train_ds.class_names
        if not "Model" in [x[0] for x in os.walk(".")]:
            os.mkdir("Model")
        class_file = open('Model/class_names.txt', 'w')
        for classes in class_names:
            class_file.write(str(classes))
            class_file.write('\n')
        class_file.close()

        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
        ])
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
        x = data_augmentation(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name="flower_vegetable_Detection_MobileNetV2")
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            metrics=["accuracy"])
        initial_epochs = iteration
        model.fit(x=train_ds,
                  epochs=initial_epochs,
                  validation_data=valid_ds)
        model.save("Model/nn.h5")
        massage = "successfully model created"
    except:
        massage = "Model is not created"
    return massage
