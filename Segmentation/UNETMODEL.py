from keras.layers import Conv2D, Input, Conv2DTranspose, BatchNormalization, Activation, Concatenate, LeakyReLU, concatenate, MaxPool2D
from keras.models import Model


# def conv_block(input,num_filters):
#     x=Conv2D(num_filters,3,padding="same")(input)
#     x=BatchNormalization()(x)
#     x=Activation("relu")(x)
#
#     x=Conv2D(num_filters,3,padding="same")(input)
#     x=BatchNormalization()(x)
#     x=Activation("relu")(x)
#
#     return x
#
# def encoder_block(input,num_filters):
#     x=conv_block(input,num_filters)
#     p=MaxPooling2D((2,2))(x)
#     return x,p
#
# def decoder_block(input,skip_features,num_filters):
#     x=Conv2DTranspose(num_filters,(2,2),strides=2,padding="same")(input)
#     x=Concatenate()([x,skip_features])
#     x=conv_block(x,num_filters)
#     return x
#
# def UNET(input_shape):
#     input=Input(input_shape)
#
#     s1,p1=encoder_block(input,64)
#     s2,p2=encoder_block(p1,128)
#     s3,p3=encoder_block(p2,256)
#     s4,p4=encoder_block(p3,512)
#
#     b1=conv_block(p4,1024)
#
#     d1=decoder_block(b1,s4,512)
#     d2=decoder_block(d1,s3,256)
#     d3=decoder_block(d2,s2,128)
#     d4=decoder_block(d3,s1,64)
#
#     outputs=Conv2D(3,1,padding="same",activation="softmax")(d4)
#
#     model=Model(input,outputs,name="U-Net")
#     return model

def UNET(inputsize=(256, 256, 3), classes=3):
    inputs = Input(shape=(inputsize))

    conv = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv1')(
        inputs)
    x = BatchNormalization()(conv)
    x = LeakyReLU()(x)
    x1 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv2')(
        x)
    x = BatchNormalization()(x1)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='MaxPool1')(x)

    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x2 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv4')(x)
    x = BatchNormalization()(x2)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), name='MaxPool2')(x)

    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x3 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv6')(x)
    x = BatchNormalization()(x3)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='MaxPool3')(x)

    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv7')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv8')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = concatenate([x, x3], axis=3)

    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv9')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv10')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = concatenate([x, x2], axis=3)

    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv11')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv12')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = concatenate([x, x1], axis=3)

    x = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv25')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', name='Conv26')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    outputs = Conv2D(classes, (1, 1), padding="same", activation='softmax', name='Outputs')(x)
    final_model = Model(inputs=inputs, outputs=outputs)
    print(final_model.summary())
    return final_model

    print(UNET())