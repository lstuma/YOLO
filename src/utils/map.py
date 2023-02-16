import numpy as np
from iou import iou

class mAP:

    def __init__(self, start: float, increment: float, stop: float, iou_threshold=.5):
        self.start = start
        self.increment = increment
        self.stop = stop
        self.iou_threshold = iou_threshold

    def __call__(self, pred_boxes: list, true_boxes: list):
        """
        pred_boxes: tuple[tuple] = [[midpoint_x, midpoint_y, width, height, c1, ..., cn],]
        true_boxes: tuple[tuple] = [[midpoint_x, midpoint_y, width, height, c1, ..., cn],]
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

        return _polygonarea(recalls, precisions)




    def _polygonarea(xs, ys):

        d = 0

        ys += [.0, .0, max(ys)]
        xs += [max(xs), .0, .0]

        for i in range(len(xs)):
            if i == len(xs)-1:
                i1 = i
                i2 = 0
            else:
                i1 = i
                i2 = i+1

            x1, y1 = xs[i1], ys[i1]
            x2, y2 = xs[i2], ys[i2]

            print(np.array([
                [x1, x2],
                [y1, y2]
            ]))

            d += abs(np.linalg.det(np.array([
                [x1, x2],
                [y1, y2]
            ])))

        return d/2






if __name__ == "__main__":
    m = mAP(.05, .05, .95)  # YO ADD TESTS LATER
    print(m.test())


