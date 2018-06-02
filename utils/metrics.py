from __future__ import print_function

import numpy as np

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

def _get(generator, model, score_threshold=0.3, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    total = generator.batch_size*(generator.size() // generator.batch)
    all_detections = [[None for i in range(generator.num_classes())] for j in range(total)]
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(total)]
    # predict_model = Model(inputs=model.get_layer("input_1").input, outputs=[model.get_layer("conv2d_{}".format(i)).output for i in [59, 67, 75]]) 
    y_pred = None

    for i in range(generator.size() // generator.batch_size):

        # Generate batch of data
        batch_data = generator.next()

        # run prediction on batch
        y = predict_model.predict_on_batch(batch_data[0][0])

        # Stacking image groups detections
        y_pred = y if y_pred is None else [np.concatenate([y_pred[j], y[j]]) for j in range(3)]

    # 3 detection layer, each having 3 anchor boxes concatenated into 9 anchor boxes    
    y_pred = np.concatenate([np.reshape(y_pred[j], (*(y_pred[j].shape[:3]), 3, *(y_pred[j].shape[3:]//3))) for j in range(3)], axis=-2)

    for image_index in range(total): 

        annotations = generator.load_annotations(image_index)

        # Find predictions with objectness > score threshold
        y_pred_mask = y_pred[image_index, ..., 4] > score_threshold

        # Each satisfied predictions of an image is an array of shape (num_satisfied_predictions, 3*(5 + num_classes))
        short = y_pred[image_index, y_pred_mask]

        # In case no detections, no need to process the below statements
        if short.shape[0] == 0:
            all_detections[image_index] = [[] for label in range(generator.num_classes())]
            all_annotations[image_index] = [annotations[annotations[:, 4] == label, :4].copy() for label in range(generator.num_classes())]
            continue

        # detections is of shape (min(max_detections, num_satisfied_predictions), 5)
        detections = np.zeros((min(max_detections, short.shape[0]), 5))

        # Sort objectness of the predictions in descending order and take the first <max_detections> values
        scores_order = np.argsort(-short[:, 4])[:max_detections]

        # Assign to detections
        detections[:, :5] = short[scores_order, :5] 
        detections[:, 5] = np.argmax(short[scores_order, 5:], axis=-1)

        for label in range(generator.num_classes()):
            all_detections[image_index][label] = detections[detections[:, -1] == label, :4].copy()
            all_annotations[image_index][label] = annotations[annotations[:, -1] == label, :4].copy()

    return all_detections, all_annotations

def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.3,
    max_detections=100,
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
    all_detections, all_annotations = _get(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
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
