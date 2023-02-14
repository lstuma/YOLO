from iou import iou

# Non max supression
def nmax(bboxes: list, threshold=0.5):
    """
        bboxes: list = [[confidence, x, y, width, height, label],...]
        threshold: floating point = IoU threshold for supression
        returns: resulting bounding boxes after non-max supression
    """

    # Sort bboxes by confidence
    bboxes.sort(key=lambda bbox: bbox[0])

    # Check all boxes
    while(bboxes):
        # List of boxes resulting after nmax supression
        true_boxes = []

        true_boxes.append(bboxes.pop(0))
        for i, bbox in enumerate(bboxes):
            # Remove bbox if it is overlapping and has same label as the selected true box
            if true_boxes[-1][5] == bbox[5] and iou(true_boxes[-1], bbox) > threshold:
                bboxes.pop(i)

    return true_boxes