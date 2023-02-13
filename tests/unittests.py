import unittest, os, sys
BASE_DIR = os.path.realpath(os.path.dirname(__file__)).rsplit('/', 1)[0]
sys.path.append(BASE_DIR)

class TestStringMethods(unittest.TestCase):

    def test_iou(self):
        # Import IoU calc function
        from src.utils.iou import iou

        print("[IoU Test] Testing IoU calculation")

        # Boxes not intersecting
        box1 = [2, 2, 2, 2]
        box2 = [7, 4, 2, 4]
        print("Expected outcome: 0,\tOutcome: ", iou(box1, box2))


        # Box2 intersecting with box1
        box1 = [2, 2, 3, 1]
        box2 = [3, 3, 1, 2]
        outcome = iou(box1, box2)
        print("Expected outcome: 0.5,\tOutcome: ", outcome)

        # Box2's left side same as box1's left side
        box1 = [2, 2, 3, 1]
        box2 = [1, 3, 1, 2]
        outcome = iou(box1, box2)
        print("Expected outcome: 0.5,\tOutcome: ", outcome)

        # Box2 inside box1
        box1 = [2, 2, 7, 5]
        box2 = [2.5, 2.5, 1, 2]
        outcome = iou(box1, box2)
        print("Expected outcome: 2,\tOutcome: ", outcome)

        # Box1 inside box2
        box1 = [2.5, 2.5, 1, 2]
        box2 = [2, 2, 7, 5]
        outcome = iou(box1, box2)
        print("Expected outcome: 2,\tOutcome: ", outcome)


# Run unittests
unittest.main()