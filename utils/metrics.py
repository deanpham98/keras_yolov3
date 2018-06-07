from PIL import Image
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda 

from .model import yolo_eval

# from keras.models import Model

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _get(generator, model, score_threshold=0.3, max_detections=20, save_path=None):

    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    boxes = None
    scores = None
    labels = None
    input_image_shape = K.placeholder(shape=(2, ))
    out_boxes, out_scores, out_labels = yolo_eval(
            yolo_outputs    = model.output,
            anchors         = generator.anchors,
            num_classes     = generator.num_classes(),
            image_shape     = input_image_shape,
            max_boxes       = max_detections,
            score_threshold = score_threshold,
            iou_threshold   = 0.5
        )
    # input_shape = Lambda(lambda x: K.reshape(x, (2, )))(input_image_shape)
    # predictions = Lambda(
    #         function  = yolo_eval,
    #         name      = 'yolo_eval',
    #         arguments = {
    #             'anchors'         : generator.anchors, 
    #             'num_classes'     : generator.num_classes(),
    #             'image_shape'     : input_shape,
    #             'max_boxes'       : max_detections,
    #             'score_threshold' : score_threshold,
    #             'iou_threshold'   : 0.5
    #         }
    #     )(model.output)

    # predict_model = Model(inputs=[model.input, input_image_shape], outputs=predictions)

    for i in range(generator.size()):

        annotations = generator.load_annotations(i)
        true_boxes_xy = (annotations[:, 0:2] + annotations[:, 2:4]) // 2
        true_boxes_wh = annotations[:, 2:4] - annotations[:, 0:2]
        annotations[:, 0:2] = true_boxes_xy
        annotations[:, 2:4] = true_boxes_wh
        image = generator.load_image(i)
        image = Image.fromarray(image[:,:,::-1])
        size = generator.input_shape
        width, height = size
        img_width, img_height = image.size
        
        scale = min(width / img_width, height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        image_data = image.resize((new_width, new_height), Image.BICUBIC)
        padded_image = Image.new('RGB', size, (128,128,128))
        padded_image.paste(image_data, ((width - new_width) // 2, (height - new_height) // 2))

        padded_image = np.array(padded_image, dtype='float32')
        padded_image = np.expand_dims(padded_image / 255., axis=0)
        
        boxes, scores, labels = K.get_session().run(
                [out_boxes, out_scores, out_labels],
                 feed_dict = {
                     model.input: padded_image,
                     input_image_shape: [image.size[1], image.size[0]],
                     K.learning_phase(): 0
                }
            )
        # inputs = [padded_image, np.expand_dims(np.array(image.size[::-1]), axis=0)]

        # boxes, scores, labels = predict_model.predict(
        #                 x          = inputs, 
        #                 batch_size = 1, 
        #                 verbose    = 0, 
        #         )

        image_boxes         = np.array(boxes)
        image_scores        = np.array(scores)
        image_labels        = np.array(labels)
        image_boxes[:, 0:2] = np.maximum(np.zeros(image_boxes[:, 1::-1].shape), np.floor(image_boxes[:, 1::-1] + 0.5))
        image_boxes[:, 2:4] = np.minimum(np.ones(image_boxes[:, 3:1:-1].shape) * generator.load_image(i).shape[1::-1], np.floor(image_boxes[:, 3:1:-1] + 0.5))
        boxes_xy            = (image_boxes[:, 0:2] + image_boxes[:, 2:4]) // 2
        boxes_wh            = image_boxes[:, 2:4] - image_boxes[:, 0:2]

        image_detections    = np.concatenate([boxes_xy, boxes_wh,np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections, all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.3,
    max_detections=20,
    save_path=None
):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_annotations = _get(generator=generator, model=model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)

    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return average_precisions

