#!/usr/bin/env python
import argparse
import os
try:
    import better_exceptions
except ImportError:
    pass
import tensorflow as tf
from .model import crnn_fn
from .data_handler import make_input_fn

from .config import Params, import_params_from_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model according to the specified config in the JSON. '
                                     'The optional arguments override the ones from the file.')
    parser.add_argument('params_file',              type=str, help='Parameters filename (JSON)')
    parser.add_argument('-o', '--output_model_dir', type=str, required=False, help='Directory for output')
    parser.add_argument('-n', '--nb_epochs',        type=int, required=False, help='Number of epochs')
    parser.add_argument('-g', '--gpu',              type=str, required=False, help="GPU 0,1 or '' ")
    args = vars(parser.parse_args())
    args = dict(filter(lambda kv: kv[1] is not None, args.items()))

    dict_params = import_params_from_json(json_filename=args.get('params_file'))
    dict_params.update(args)
    parameters = Params(**dict_params)

    model_params = {
        'Params': parameters,
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True

    # Config estimator
    est_config = tf.estimator.RunConfig().replace(
        keep_checkpoint_max=200,
        save_checkpoints_steps=parameters.save_interval,
        session_config=config_sess,
        save_checkpoints_secs=None,
        save_summary_steps=1000,
        model_dir=parameters.output_model_dir
    )

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,     # Create the pipeline
                                       params=model_params,
                                       model_dir=parameters.output_model_dir,
                                       config=est_config
                                       )

    estimator.evaluate(input_fn=make_input_fn(parameters.tfrecords_eval,
                                              parameters.eval_batch_size,
                                              parameters.input_shape,
                                              data_augmentation=False,
                                              dynamic_distortion=False,
                                              repeat=False)
                      )
