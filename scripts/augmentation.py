from keras.preprocessing.image import ImageDataGenerator
class aug:
    # Training generator
    @staticmethod
    def train_augment(train_dir):
        LR = 1e-3
        height=150
        width=150
        channels=3
        seed=1337
        batch_size = 64
        num_classes = 2
        epochs = 5
        data_augmentation = True
        num_predictions = 20

      
        train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(height,width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    shuffle=True,
                                                    class_mode='categorical')
        return train_generator

        # Test generator
    @staticmethod
    def test_augment(test_dir):
        LR = 1e-3
        height=150
        width=150
        channels=3
        seed=1337
        batch_size = 64
        num_classes = 2
        epochs = 5
        data_augmentation = True
        num_predictions = 202
        test_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(height,width), 
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  shuffle=True,
                                                  class_mode='categorical')
        return validation_generator

        