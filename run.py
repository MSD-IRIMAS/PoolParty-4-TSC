import argparse
import time

import numpy as np
import tensorflow as tf

import classifiers.fcn
import classifiers.resnet
import classifiers.inception
import classifiers.lite
import constants
import data
import layers
import logger

# Log information
print('TF ', tf.__version__)
print('GPU', tf.config.list_physical_devices('GPU'))

# Unique ID
exp_id = str(time.time()).replace('.', '') # int(time.time()*10)

# Parse arguments
model_mapping = {
    'fcn': classifiers.fcn.get_model,
    'resnet': classifiers.resnet.get_model,
    'inception': classifiers.inception.get_model,
    'lite':  classifiers.lite.get_model,
}
reduce_mapping = {
    'gap':     (tf.keras.layers.GlobalAveragePooling1D,),
    'gmp':     (tf.keras.layers.GlobalMaxPooling1D,),

    'rnn64':   (tf.keras.layers.SimpleRNN, 64),
    'lstm64':  (tf.keras.layers.LSTM, 64),
    'gru64':   (tf.keras.layers.GRU, 64),
    'rnn128':  (tf.keras.layers.SimpleRNN, 128),
    'lstm128': (tf.keras.layers.LSTM, 128),
    'gru128':  (tf.keras.layers.GRU, 128),

    'ppv':     (layers.PPV,),        # PPV = SoftPPV
    'ppvH':    (layers.HardPPV,),
    'ppvS':    (layers.SoftPPV,),
    'ppvHste': (layers.HardPPVSTE,),
    'mpv':     (layers.MPV,),        # MPV = HardMPV
    'mpvH':    (layers.HardMPV,),
    'mpvS':    (layers.SoftMPV,),
    'mipv':    (layers.MIPV,),       # MIPV = SoftMIPV
    'mipvS':   (layers.HardMIPV,),
    'mipvH':   (layers.SoftMIPV,),

    'map2':  (layers.MultiAveragePooling1D, 2),
    'map4':  (layers.MultiAveragePooling1D, 4),
    'map8':  (layers.MultiAveragePooling1D, 8),
    'map16': (layers.MultiAveragePooling1D, 16),
    'mmp2':  (layers.MultiMaxPooling1D, 2),
    'mmp4':  (layers.MultiMaxPooling1D, 4),
    'mmp8':  (layers.MultiMaxPooling1D, 8),
    'mmp16': (layers.MultiMaxPooling1D, 16),
}
data_mapping = {
    'tsc_ucr': constants.UNIVARIATE_DATASET_NAMES_2018,
    'tsc_ucr_s': constants.UNIVARIATE_DATASET_NAMES_SHORTLIST,
    'tsc_ucr_m': constants.UNIVARIATE_DATASET_NAMES_SHORTLIST_M,
    'mtsc_uea': constants.MULTIVARIATE_DATASET_NAMES_2018
}
parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--model',  type=str, required=True,
                    choices=model_mapping.keys())
parser.add_argument('--reduce', type=str, required=True, nargs='+',
                    choices=reduce_mapping.keys())
parser.add_argument('--dataset', type=str, required=True, nargs='+',
                    choices=data_mapping.keys())
parser.add_argument('--results', type=str, default='results',
                    help='Folder to save results')
args = parser.parse_args()
# Define model and reduce layer
model_name = args.model
reduce_names = args.reduce
model_builder = model_mapping.get(model_name)
datasets = []
for d_name in args.dataset:
    datasets += data_mapping.get(d_name)
results = args.results
# Columns to log
columns = {'model': str, 'reduce': str, 'dataset': str, 'train_time_s': int,
           'accuracy': float, 'f1_score': float, 'loss': float,
           'train_accuracy': float, 'train_f1_score': float, 'train_loss': float}

# Logger
logger = logger.CSV(f'{results}/EXP_{args.model}_{"_".join(reduce_names)}_{exp_id}.csv', columns)


def train(x_train, y_train, n_classes):
    reduce = []
    for reduce_name in reduce_names:
        r = reduce_mapping.get(reduce_name)
        f = r[0]
        if len(r) > 1:
            reduce.append(f(*(r[1:])))
        else:
            reduce.append(f())

    if any(name in [f'map{i}' for i in range(100)] + [f'mmp{i}' for i in range(100)] for name in reduce_names):
        input_shape = (x_train.shape[-2], x_train.shape[-1])
    else:
        input_shape = (None, x_train.shape[-1])

    model = model_builder(n_classes=n_classes, reduce=reduce, input_shape=input_shape)
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.F1Score(average='micro')])
    
    model.save(f'bestweights/{exp_id}.keras')
    
    # callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'bestweights/{exp_id}.keras', monitor='loss', save_best_only=True)
    callbacks = [reduce_lr, model_checkpoint]

    # train parameters
    batch_size = 16
    nb_epochs = 2000
    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

    # run train
    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, callbacks=callbacks, verbose=0)
    # rollback to best model on train loss
    model = tf.keras.models.load_model(f'bestweights/{exp_id}.keras')

    return model


for dataset_name in datasets:
    x_train, y_train, x_test, y_test = data.load_dataset(dataset_name)
    length_ts = int(x_train.shape[1])
    n_classes = int(y_train.shape[1])

    t0 = time.time()
    model = train(x_train, y_train, n_classes)
    t1 = time.time()
    
    train_metrics = model.evaluate(x_train, y_train, return_dict=True, verbose=0)
    metrics = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
    
    print(f"{metrics['categorical_accuracy']:.4f} {metrics['f1_score']:.4f} {metrics['loss']:.4f} {dataset_name}")

    logger.add({'model': model_name, 'reduce': "_".join(reduce_names), 'dataset': dataset_name, 'train_time_s': int(t1 - t0),
           'accuracy': metrics['categorical_accuracy'], 'f1_score': metrics['f1_score'], 'loss': metrics['loss'],
           'train_accuracy': train_metrics['categorical_accuracy'], 'train_f1_score': train_metrics['f1_score'], 'train_loss': train_metrics['loss']})
