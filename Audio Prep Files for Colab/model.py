# Build Model

def get_model(r,c) :

    # hyper-parameters
    n_filters = 64
    filter_width = 3
    dilation_rates = [2**i for i in range(8)] 

    history_seq = Input(shape=(r, c))
    x = history_seq

    skips = []
    count = 0
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate, activation='relu', name="conv1d_dilation_"+str(dilation_rate))(x)
        
        x = BatchNormalization()(x)


    out = Conv1D(32, 16, padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('tanh')(out)
    out = GlobalMaxPool1D()(out)

    model = Model(history_seq, out)
    model.compile(loss='mse', optimizer='adam')

    input1 = Input((r,c), name="Anchor Input")
    input2 = Input((r,c), name="Positive Input")
    input3 = Input((r,c), name="Negative Input")

    anchor = model(input1)
    positive = model(input2)
    negative = model(input3)


    concat = concatenate([anchor, positive, negative], axis=1)

    siamese = Model([input1, input2, input3], concat)


    siamese.compile(optimizer='adam', loss=triplet_loss)

    print(siamese.summary())

    return model, siamese