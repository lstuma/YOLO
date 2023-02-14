
# Intersection over Union
def iou(box1, box2, anchor_type="midpoint"):
    """
        box1, box2: tuple = [midpoint_x, midpoint_y, width, height]
        anchor_type: str = "midpoint" or "corners"
        returns: IoU of the two boxes
    """

    # Fix this later
    if anchor_type not in ["midpoint", "corners"]:
        raise NotImplementedError("Anchor type not implemented")

    if anchor_type == "midpoint":
        # Convert from midpoint boxes to corner boxes for easier calculation
        for box in [box1, box2]:
            # Calculate upper left corner
            box[0] = box[0] - box[2]/2
            box[1] = box[1] - box[3]/2
            # Calculate lower right corner
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]

    # Calculate IoU
    iou_dim = [0, 0]
    # Check cases:
    for i in range(2):
        j = i+2
        # box1 inside box2
        if box1[i] >= box2[i] and box1[j] <= box2[j]:
            #print("box1 inside box2")
            iou_dim[i] = box1[j]-box1[i]
        # box2 inside box1
        elif box2[i] >= box1[i] and box2[j] <= box1[j]:
            #print("box2 inside box1")
            iou_dim[i] = box2[j]-box2[i]
        # box1 has left side in box2
        elif box2[i] <= box1[i] <= box2[j]:
            #print("box1 has left side in box2")
            iou_dim[i] = box2[j]-box1[i]
        # box2 has left side in box1
        elif box1[i] <= box2[i] <= box1[j]:
            #print("box2 has left side in box1")
            iou_dim[i] = box2[i]-box1[j]
        # boxes do not intersect
        #elif box2[j] <= box1[i] or box2[i] >= box1[j]:
            #print("no intersection")

    return abs(iou_dim[0] * iou_dim[1])
