from __future__ import print_function
from __future__ import absolute_import

import os
import time
import warnings

import numpy as np
import keras
import tensorflow as tf

from .metrics import evaluate


class SaveWeights(keras.callbacks.Callback):
	def __init__ (self, infer_model):
		self.infer_model  = infer_model
		self.weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weights')

		super(SaveWeights, self).__init__()

	def on_epoch_end(self, epoch, logs=None):
		self.infer_model.save_weights(os.path.join(self.weights_path, 'epoch_' + str(epoch) + time.strftime("_%d_%m_%Y_%H_%M_%S") + '.weights'))


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.
    # Arguments
        callback: callback to wrap.
        model: model to use when executing callbacks.
    # Example
        ```python
        model = keras.models.load_model('model.h5')
        model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
        ```
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)

class Metrics(keras.callbacks.Callback):
    def __init__(
        self,
    	generator,
    	iou_threshold     = 0.5,
    	score_threshold   = 0.3,
    	max_detections    = 100,
    	save_path         = None,
    	tensorboard       = None,
        filepath          = None,
        monitor           = 'mAP',
        mode              = 'max',
        save_best_only    = True,
        save_weights_only = True,
        period            = 1,
    	verbose           = 1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        assert generator.training == False, "training property not set to False"

        self.generator              = generator
        self.iou_threshold          = iou_threshold
        self.score_threshold        = score_threshold
        self.max_detections         = max_detections
        self.save_path              = save_path
        self.tensorboard            = tensorboard
        self.verbose                = verbose
        self.filepath               = filepath
        self.monitor                = monitor
        self.mode                   = mode
        self.save_best_only         = save_best_only
        self.save_weights_only      = save_weights_only
        self.period                 = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Metrics checkpoint mode %s is unknown, '
                          'fallback to max mode.' % (mode),
                          RuntimeWarning)
            mode = 'max'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or \
               self.monitor == 'mAP' or \
               self.monitor.startswith('fmeasure'):

                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        super(Metrics, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
                generator       = self.generator,
		model           = self.model,
		iou_threshold   = self.iou_threshold,
		score_threshold = self.score_threshold,
		max_detections  = self.max_detections,
		save_path       = self.save_path
       )

        self.mAP = sum(average_precisions.values()) / len(average_precisions)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mAP
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mAP

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch+1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Skipping. Can save best model only with %s available' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to \%0.5f, saving model to %s' % (epoch+1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from \ %0.5f'  % (epoch+1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' \
                            % (epoch+1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
        print()
        if self.verbose == 1:
            for label, average_precision in average_precisions.items():
                print(self.generator.label_to_name(label), '{:.4f}'.format(average_precision))
