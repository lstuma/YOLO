from utils.iou import iou

# Non max supression
def nmax(bboxes: list, threshold=0.5):
    """
        bboxes: list = [[confidence, x, y, width, height, label],...]
        threshold: floating point = IoU threshold for supression
        returns: resulting bounding boxes after non-max supression
    """

    # Sort bboxes by confidence
    bboxes.sort(key=lambda bbox: 1-bbox[0])

    # List of boxes resulting after nmax supression
    true_boxes = []


    # Check all boxes
    while(bboxes):
        true_boxes.append(bboxes.pop(0))
        for bbox in bboxes.copy():
            # Remove bbox if it is overlapping and has same label as the selected true box
            if true_boxes[-1][5] == bbox[5] and iou(true_boxes[-1], bbox) > threshold:
                bboxes.remove(bbox)

    return true_boxes