import cv2
import supervision as sv
import os
from polygon_test import PolygonTest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CrowdDetection:

    def __init__(self, model, line_zones, zones):

        self.model = model
        self.line_zones = line_zones
        self.zones = zones
        self.detections = None
        self.count = None
        self.frame = None
        self.crowd_detection_flag = False
        self.elastic_list = []
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def plots(self):
        try:
            cv2.putText(
                img=self.frame,
                text=f"Crowd Detected | Person_Count: {self.count}",  # Shortened text
                org=(400, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Changed font style
                fontScale=1,  # Adjust font size
                color=(0, 0, 255),
                thickness=2  # Adjust thickness
            )
            cv2.polylines(self.frame, [self.zones], True, (0, 0, 255), 4)
        except Exception as er:
            print(er)

    def polygon_test(self):
        try:
            intersect, self.count = PolygonTest(self.detections, self.line_zones).point_polygon_test()
            if intersect and self.count > 5:
                self.plots()
                self.crowd_detection_flag = True
            else:
                self.crowd_detection_flag = False
        except Exception as er:
            print(er)

    def predict(self, q_img):
        try:
            self.frame = q_img.get()
            result = self.model(source=self.frame, conf=0.5, classes=0, verbose=False)
            result = result[0]

            self.detections = sv.Detections.from_ultralytics(result)

            if self.detections:
                labels = ["{}".format(result.names[cls_id]) for xyxy, mask, cfg, cls_id, tracker_id, cls_name in self.detections ]
                self.box_annotator.annotate(scene=self.frame, detections=self.detections)
                self.label_annotator.annotate(scene=self.frame, detections=self.detections, labels=labels)
                self.polygon_test()

                return self.frame
            else:
                return self.frame
        except Exception as er:
            print(er)


