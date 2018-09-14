#Originated by autokeras github
from abc import abstractmethod
from sklearn.metrics import accuracy_score, mean_squared_error

class Metric:

    @classmethod
    @abstractmethod
    def higher_better(cls):
        pass

    @classmethod
    @abstractmethod
    def compute(cls, prediction, target):
        pass

    @classmethod
    @abstractmethod
    def evaluate(cls, prediction, target):
        pass

class Accuracy(Metric):
    @classmethod
    def higher_better(cls):
        return True

    @classmethod
    def compute(cls, prediction, target):
        prediction = list(map(lambda x: float(x.argmax()), prediction))
        target = list(map(lambda x: float(x.argmax()), target))
        return cls.evaluate(prediction, target)

    @classmethod
    def evaluate(cls, prediction, target):
        return accuracy_score(prediction, target)


class MSE(Metric):
    @classmethod
    def higher_better(cls):
        return False

    @classmethod
    def compute(cls, prediction, target):
        return cls.evaluate(prediction, target)

    @classmethod
    def evaluate(cls, prediction, target):
        return mean_squared_error(prediction, target)

class IOU(Metric):
    @classmethod
    def higher_better(cls):
        return True

    @classmethod
    def compute(cls, prediction, target):
        return cls.evalute(prediction, target)

    @classmethod
    def evaluate(cls, anchors, gt_boxes):
	len_anchors = anchors.shape[0]
	len_gt_boxes = gt_boxes.shape[0]
	anchors = np.repeat(anchors, len_gt_boxes, axis=0)
	gt_boxes = np.vstack([gt_boxes] * len_anchors)

	y1 = np.maximum(anchors[:, 0], gt_boxes[:, 0])
	x1 = np.maximum(anchors[:, 1], gt_boxes[:, 1])
	y2 = np.minimum(anchors[:, 2], gt_boxes[:, 2])
	x2 = np.minimum(anchors[:, 3], gt_boxes[:, 3])

	y_zeros = np.zeros_like(y2.shape)
	x_zeros = np.zeros_like(x2.shape)

	intersect = np.maximum((y2 - y1), y_zeros) * np.maximum((x2 - x1), x_zeros)

	unit = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]) + \
	       (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) - intersect

	return (intersect / unit).reshape(len_anchors, len_gt_boxes)

class AP(Metric):
    @classmethod:
    def higher_better(cls):
        return True

    @classmethod
    def compute(cls, recall, precision):
        return cls.evalute(recall, precision)

    @classmethod
    def evaluate(cls, recall, precision):
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


