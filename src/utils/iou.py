
# Intersection over Union
def iou(box1, box2, anchor_type="midpoint"):
    """
        box1, box2: tuple = [midpoint_x, midpoint_y, width, height]
    """
    # Fix this later
    if anchor_type != "midpoint":
        raise NotImplementedError("Only midpoint implemented")

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
    print(box1, box2)
    # Check cases:
    for i in range(2):
        j = i+1
        # box1 inside box2
        if box1[i] > box2[i] and box1[j] < box2[j]:
            print("box1 inside box2")
            iou_dim[i] = box1[j]-box1[i]
        # box1 has left side in box2
        elif box2[i] < box1[i] < box2[j]:
            print("box1 has left side in box2")
            iou_dim[i] = box2[j]-box1[i]
        # box2 has left side in box1
        elif box1[i] < box2[i] < box1[j]:
            print("box2 has left side in box1")
            iou_dim[i] = box2[i]-box1[j]
        # box2 inside box1
        elif box2[i] > box1[i] and box2[j] < box1[j]:
            print("box2 inside box1")
            iou_dim[i] = box2[j]-box2[i]
        # boxes do not intersect
        elif box2[j] < box1[i] or box2[i] > box1[j]:
            print("no intersection")



    

    # iou_width = box1[2]-box2[0] if box1[2] > box2[0] else box2[2]-box1[0]
    # iou_height = box1[3]-box2[1] if box1[3] > box2[1] else box2[3]-box1[1]

    print("--> iou_width: ", iou_dim[0], ", iou_height: ", iou_dim[1])

    return iou_dim[0] * iou_dim[1]


if __name__ == '__main__':
    box1 = [2, 2, 2, 2]
    box2 = [7, 4, 2, 4]
    print("Expected outcome: 0,\tOutcome: ", iou(box1, box2))


    box1 = [2, 2, 3, 1]
    box2 = [3, 3, 1, 2]
    print("Expected outcome: 0.5,\tOutcome: ", iou(box1, box2))

    box1 = [2, 2, 3, 1]
    box2 = [1, 3, 1, 2]
    print("Expected outcome: 0.5,\tOutcome: ", iou(box1, box2))

    box1 = [2, 2, 7, 5]
    box2 = [2.5, 2.5, 1, 2]
    print("Expected outcome: 2,\tOutcome: ", iou(box1, box2))