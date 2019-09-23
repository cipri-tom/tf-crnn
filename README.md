# Type-aware CRNN

This repository contains the code for our ICDAR'19 paper _Field typing for improved recognition on handwritten forms_ (links to come).

It is built on top of [tf_crnn](https://github.com/solivr/tf-crnn), which is a tensorflow-estimator version of the orignal Convolutional Recurrent Neural Network (CRNN) (original [paper](http://arxiv.org/abs/1507.05717) and [code](https://github.com/bgshih/crnn))

We used it to sucessfully parse European Accident Statements in French, but it can be easily adapted to any use case that demands transcription of heterogeneous handwritten context.

## Installation

This package requires `python3` and TensorFlow. While CPU version could work, the high computational demands mean that the GPU version is necessary for training.

Currently it has been tested only under `python v3.6.6` since this is the only `py3.6` version that official TensorFlow distribution supports. More compatibility to come.

You may already have a working version of tensorflow, which is why we do not try to automatically install it for you. If you do not, please install it before installing this package:


```bash
$ pip install tensorflow-gpu
```

Then, you can install this package. From the root directory, run:

```bash
$ pip install -r requirements.txt
```

This will symlink the package in the python installation directory, so that if you make modifications, they will instantly be available. Note that installation is not necessary for a simple run, but then all commands should be run from the root directory (because Python automatically discovers packages in the current directory).

## Training

### Synthetic training data
We have created a dataset as mentioned in section III-B of our paper. It is available [here](https://drive.google.com/uc?id=1upVeN8a_sHT67rYupPs64KyIKC9_qYdO&export=download). Download and extract the zip file to a directory of your choosing, for example `~/type-aware-crnn/data/`. Then, place this path with a glob pattern in the `model_params.json` config file (see below).

### Evaluation data
Unfortunately, we cannot provide this as it contains sensitive information. It should be in the same format as the training data, namely a tfrecords file where each example has the followin `feature_spec`:
```python
feature_spec = {
    'image_raw': tf.FixedLenFeature([], tf.string), # the PNG/JPG bytes
    'label': tf.FixedLenFeature([], tf.string),     # the transcription
    'corpus': tf.FixedLenFeature([],tf.int64),      # the type id
}
```
If you are not used to generating tfrecords data, open an issue and we will support you in converting your existing dataset.

### Training configuration

To train, you need to provide a parameters file. An example one is `model_params.json`. You should modify at least the following paths in there:

*  `output_model_dir`:  path to where the model will save weights, e.g. `"~/type-aware-crnn/models/french_model"`. Note that if it already contains model checkpoints from a previous training, the training will continue.
* `tfrecords_train`: path to training data, e.g. `"~/type-aware-crnn/data/train/2M_dilgrad_train_batch_*.tfrecords"` (if you are using our synthetic data)
*  `tfrecords_eval`: path to evaluation data e.g. `"~/type-aware-crnn/data/eval/*.tfrecords"`

Then, you can start training with:
```bash
python -m tf_crnn.train <path_to_model_params.json>
```

You can quickly modify the output directory, the GPU being used or the number of epochs by providing optional parameters to the script, which override the ones in the JSON file. See `python -m tf_crnn.train -h`

---

Documentation for other important parameters:
 - `input_shape`: all input images are transformed to this shape to be able to form batches. Note that we the image is replicated horizontally to avoid filling the batch with white space (default: `[32, 256]`)
 - `num_corpora`: how many different types to use (default: `10`)
 - `gpu`: the ID of the GPU to be used (single GPU model)
 - `n_epochs`: how many epochs to train for
 - `learning_rate`: self explaining
 - `learning_rate_decay` : if non-zero, then an exponentially decreasing learning rate is used with this decay rate (default: `1e-4`)
 - `learning_rate_steps` : indicates when to decrease the LR, if the decay is defined
 - `train_batch_size`: size of the batch for training. The bigger the batch, the more memory you need
 - `eval_batch_size`: size of the batch for eval
 - `save_interval`: how often to save a checkpoint
 - `keep_prob`: when using dropout, how many to keep
 - `evaluate_every_epoch`: how often to evaluate the model
 - `alphabet`: one of `['digits_only', 'letters_only', 'letters_digits', 'letters_extended', 'letters_digits_extended']`. Since we train for French data, note that this contains accented characters. See `config.py`.
 - `alphabet_decoding`: `same` or one of `alphabet`s. When decoding the predicted codes, we can choose to get a different mapping, for example to translate the codes for upper-case characters into their lower-case equivalents.
 - `train_cnn`: `1` if you want to train the whole network; `0` if you want to train just the LSTM part.
 - `nb_logprob` and `top_paths`: these control the output of the model. `nb_logprob` represents the width of the beam in the search algorithm. However, the model will only output the first `top_paths` out of these.
 - `dynamic_distortion`: Whether to apply elastic distortion as a data augmentation step. Enabling this (`true`) will demand more CPU resources and can slow down training.





## Contents

* `model.py` : definition of the model
* `data_handler.py` : functions for data loading, preprocessing and data augmentation
* `config.py` : `class Params` manages parameters of model and experiments
* `decoding.py` : helper function to transform characters to words
* `train.py` : script to launch for training the model, more info on the parameters and options inside
* `export_model.py`: script to export a model once trained, i.e for serving (prediction)
* Extra : `hlp/numbers_mnist_generator.py` : generates a sequence of digits to form a number using the MNIST database
* Extra : `hlp/csv_path_convertor.py` : converts a csv file with relative paths to a csv file with absolute paths
