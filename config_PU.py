class DefaultConfigs(object):
    seed =66        #666 #100  #66     #20
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.00001
    # training parameters
    train_epoch = 200
    test_epoch = 1
    BATCH_SIZE_TRAIN = 64
    norm_flag = True
    gpus = '0'
    data = 'Houston2013'  # PaviaU-9-103-0.95-666 / Indian-16-200-0.9  / Houston2018-21  / Houston2013-15-0.95-20-60
    num_classes = 15
    patch_size = 15
    pca_components = 32
    test_ratio = 0.95
    # model
    embed_dim = 32
    d_state = 16
    pos = False
    cls = False
    # 3DConv parameters
    test_freq=1
    conv3D_channel = 32
    conv3D_kernel_1 = (5, 5, 5)  #(5, 5, 5)
    dim_patch = patch_size - conv3D_kernel_1[1] + 1  # 8
    dim_linear_1 = pca_components - conv3D_kernel_1[0] + 1  # 28
    # paths information
    checkpoint_path = ('./' + "checkpoint/" + data + '/' + '_TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN)\
                      + '/PatchSize' + str(patch_size) + '_TestRatio' + str(test_ratio) \
                      + '/'  + 'Depth' + str(depth) + '_embed' + str(embed_dim) + '_dstate' + str(d_state) + '_ratio' + str(ssm_ratio)
                      + '_3Dconv' + str(conv3D_channel) + '&' + str(conv3D_kernel_1) + '/')
    logs = checkpoint_path

    def __init__(self):
        for name, val in self.__class__.__dict__.items():
            if not name.startswith("__") and not callable(val):
                setattr(self, name, val)

config = DefaultConfigs()

