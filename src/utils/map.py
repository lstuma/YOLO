import numpy
from iou import iou

class chmAP:

    def __init__(self, start: float, increment: float, stop: float, iou_threshold=.5):
        self.start = start
        self.increment = increment
        self.stop = stop
        self.iou_threshold = iou_threshold

    def __call__(self, pred_boxes: list, true_boxes: list):
        """
        pred_boxes: tuple[tuple] = [[midpoint_x, midpoint_y, width, height, confidence],]
        true_boxes: tuple[tuple] = [[midpoint_x, midpoint_y, width, height],]
        """

        combinations = []
        for pred in pred_boxes:
            box = None
            i = None
            for true in true_boxes:
                box = true
                i = iou(pred, true)
                if i > self.iou_threshold:
                    break
            combinations.append([i, pred[-1], box is not None,])  # iou, confidence, TP/FP

        combinations.sort(key=lambda k: k[1], reverse=True)

        n_tp = len(tuple(i for i in combinations if i[2]))

        precisions, recalls = [], []

        for j in range(len(combinations)):
            i, confidence, tp_fp = combinations[j]

            n_tp_j = len(tuple(i for i in combinations[:j] if i[2]))

            precision = n_tp_j / (j - 1)
            recall    = n_tp_j / n_tp

            combinations[j].append(precision)
            combinations[j].append(recall)

            precisions.append(precision)
            recalls.append(recall)







        return


if __name__ == "__main__":
    m = mAP(.05, .05, .95)  # YO ADD TESTS LATER
