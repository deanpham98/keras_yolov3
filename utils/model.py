"""YOLO_v3 Model Defined in Keras."""

import os
from functools import wraps, reduce


import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2



def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)

    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    return compose(
              DarknetConv2D(*args, **no_bias_kwargs),
              BatchNormalization(),
              LeakyReLU(alpha=0.1)
           )

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3))
            )(x)
        x = Add()([x,y])

    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)

    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1))
        )(x)

    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1))
        )(x)

    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))

    # First detection layer
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    # Upsampling
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])

    # Second detection layer
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    # Upsampling
    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])

    # Third detection layer
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])


def yolo_training(
         num_classes  = 3,
         input_shape  = (416, 416),
         anchors      = np.array(((10,13), (16,30), (33,23), (30,61), (62,45),
                           (59,119), (116,90), (156,198), (373,326))),
         model_path   = None,
         weights_path = None,
         freeze_body  = True
   ):

    # Assume the weights path is available and exists
    assert weights_path is not None, "Weights path is not provided"

    infer_model = None
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors) // 3
    y_true = [Input(shape=(h//32, w//32, num_anchors, num_classes+5)),
                  Input(shape=(h//16, w//16, num_anchors, num_classes+5)),
                  Input(shape=(h//8, w//8, num_anchors, num_classes+5))]

    if model_path is not None:
        # Check if model path exists

        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_path)

        assert os.path.exists(model_path), "Model path does not exist"
        # Load uncompiled infer model
        infer_model = load_model(model_path, compile=False)
    else:
        # Input layer 2, 3, 4 which are the offsets from ground truths

        # Rebuild the infer model
        infer_model = yolo_body(image_input, num_anchors, num_classes)

    weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weights', weights_path)

    assert os.path.exists(weights_path), "Weights path does not exist"

    # Load weights into infer model by name and skip mismatch
    infer_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # Freeze the weights of some layers for fine-tuning
    if freeze_body:
        for layer_index in range(len(infer_model.layers) - 3):
            infer_model.layers[layer_index].trainable = False

    # Custom output layer for training with output as the value of objective function which we want to optimize
    loss_layer = Lambda(
                    function     = yolo_loss,
                    output_shape = (1,),
                    name         = 'yolo_loss',
                    arguments    = {'anchors': anchors, 'num_classes': num_classes}
                )([*infer_model.output, *y_true])
    
    # Build the training model
    model = Model(
               inputs  = [infer_model.input, *y_true], 
               outputs = loss_layer
           )

    # Return both the infer model and the training model
    return infer_model, model


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters.
    
    Arguments:
    - feats: outputs of detection layers of YOLOv3
    - anchors: anchor boxes at a specified detection layer
    """

    # Number of anchor boxes at this detection layer
    num_anchors = len(anchors)

    # Reshape/Expand dimensions to (batch, height, width, num_anchors, 2).
    # Each anchor box parameters are the offset widths and heights only.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # Outputs of each detection layer has shape (batch_size, height, width, num_anchors * (5 + num_classes))
    # grid_shape tells the shape (height, width) of the detection layer
    # grid_shape is a tensor of shape (None, None)
    grid_shape = K.shape(feats)[1:3] 

    # 
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])

    # grid tensor has shape (grid_shape[0], grid_shape[1], 1, 2)
    # grid tensor represents offsets of grid cells top left corner from the image top left corner
    grid = K.concatenate([grid_x, grid_y])

    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Relative (w.r.t grid cell top left corner) positional coordinates of predicted box
    box_xy = K.sigmoid(feats[..., :2])

    # # Relative (w.r.t grid cell sizes) size coordinates of predicted box 
    box_wh = K.exp(feats[..., 2:4])

    # Objectness/Confidence score is computed using logistic function.
    box_confidence = K.sigmoid(feats[..., 4:5])

    # Class probabilities are computed using logistic function ()
    box_class_probs = K.sigmoid(feats[..., 5:])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (box_xy + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = box_wh * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    
    if calc_loss == True:
    	return grid, feats, box_xy, box_wh, anchors_tensor
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
    													  feats       = feats,
        												  anchors     = anchors, 
        												  num_classes = num_classes, 
        												  input_shape = input_shape
    												  )
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(
        yolo_outputs,
        anchors,
        num_classes,
        image_shape,
        max_boxes       = 20,
        score_threshold = 0.6,
        iou_threshold   = 0.5
    ):

    """Evaluate YOLO model on given input and return filtered boxes."""
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    
    boxes = []
    
    box_scores = []
    
    for l in range(3):
    
        _boxes, _box_scores = yolo_boxes_and_scores(
                                    feats       = yolo_outputs[l],
                                    anchors     = anchors[anchor_mask[l]], 
                                    num_classes = num_classes, 
                                    input_shape = input_shape, 
                                    image_shape = image_shape,
                              )

        boxes.append(_boxes)

        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)

    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold

    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
    
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
    
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        
        nms_index = tf.image.non_max_suppression(
                                boxes           = class_boxes, 
                                scores          = class_box_scores, 
                                max_output_size = max_boxes_tensor, 
                                iou_threshold   = iou_threshold
                            )

        class_boxes = K.gather(class_boxes, nms_index)
        
        class_box_scores = K.gather(class_box_scores, nms_index)

        classes = K.ones_like(class_box_scores, 'int32') * c
        
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou



def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(T, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''

    # YOLOv3 outputs are 3 numpy array of shape (batch_size, height, width, num_anchors * (5 + num_classes))
    yolo_outputs = args[:3]
    
    # Ground truth is a list of 3 numpy array of shape (batch_size, height, width, num_anchors, 5 + num_classes)
    y_true = args[3:]

    # Anchors size decrease during top-down upsampling path way
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    # Input shape is 32 times more than the first detection layer shape
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    
    # Shapes of each detection layer
    grid_shapes = [K.cast(K.shape(yolo_outputs[layer])[1:3], K.dtype(y_true[0])) for layer in range(3)]
    
    # Initialize loss
    xy_loss = 0
    wh_loss = 0
    confidence_loss = 0
    class_loss = 0
    loss = 0
    
    batch_size = K.shape(yolo_outputs[0])[0]

    for layer in range(3):

        # True objectness score/confidence (either 1 or 0) of each anchor-offset at each grid cell
        object_mask = y_true[layer][..., 4:5]

        # True one-hot encoded class probabilities of each ground truth box
        true_class_probs = y_true[layer][..., 5:]

        grid, y_pred, pred_xy, pred_wh, anchors_tensor = yolo_head(
                                                                  feats       = yolo_outputs[layer],
                                                                  anchors     = anchors[anchor_mask[layer]], 
                                                                  num_classes = num_classes, 
                                                                  input_shape = input_shape,
                                                                  calc_loss   = True
                                                              )

        pred_box = K.concatenate([pred_xy, pred_wh])

        true_xy = y_true[layer][..., :2] * grid_shapes[layer][::-1] - grid

        true_wh = K.log(y_true[layer][..., 2:4] * input_shape[::-1] / anchors_tensor)

        # Avoid log(0) = -inf
        true_wh = K.switch(object_mask, true_wh, K.zeros_like(true_wh))

        box_loss_scale = 2 - y_true[layer][..., 2:3] * y_true[layer][..., 3:4]

        ignore_mask = tf.TensorArray(
                            dtype        = K.dtype(y_true[0]), 
                            size         = 1, 
                            dynamic_size = True
                        )

        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):

            true_box = tf.boolean_mask(
                              tensor = y_true[layer][b, ..., 0:4],
                              mask   = object_mask_bool[b, ..., 0]
                          )

            iou = box_iou(pred_box[b], true_box)

            best_iou = K.max(iou, axis=-1)

            ignore_mask = ignore_mask.write(
                                        index = b, 
                                        value = K.cast(best_iou < ignore_thresh, K.dtype(true_box))
                                      )

            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(
                             cond      = lambda b, *args: b < batch_size, 
                             body      = loop_body, 
                             loop_vars = [0, ignore_mask]
                         )

        ignore_mask = ignore_mask.stack()

        ignore_mask = K.expand_dims(ignore_mask, axis=-1)

        xy_loss += K.sum(object_mask * box_loss_scale * K.binary_crossentropy(true_xy, y_pred[..., 0:2]))

        wh_loss += K.sum(object_mask * box_loss_scale * 0.5 * K.square(true_wh - y_pred[..., 2:4]))

        # log_weight = ignore_mask + (ignore_mask - 1) * object_mask
        #confidence_loss = (1 - object_mask) * y_pred[..., 4] + \
        #									log_weight * K.log(1 + K.exp(0 - K.abs(y_pred[..., 4]))) + \
        #													   K.relu(0 - y_pred[..., 4])

        confidence_loss += K.sum(object_mask * K.binary_crossentropy(object_mask, y_pred[..., 4:5], from_logits=True) + \
                                (1 - object_mask) * K.binary_crossentropy(object_mask, y_pred[..., 4:5], from_logits=True) * ignore_mask)

        # confidence_loss = tf.nn.weighted_cross_entropy_with_logits(object_mask, y_pred[..., 4:5], ignore_mask)

        class_loss += K.sum(object_mask * K.binary_crossentropy(true_class_probs, y_pred[..., 5:], from_logits=True))

    tf.summary.scalar('xy_loss', xy_loss)
    tf.summary.scalar('wh_loss', wh_loss)
    tf.summary.scalar('confidence_loss', confidence_loss)
    tf.summary.scalar('class_loss', class_loss)

    loss += xy_loss + wh_loss + confidence_loss + class_loss
    tf.summary.scalar('loss', loss)

    return loss
