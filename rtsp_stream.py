import os
import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from multiprocessing.pool import ThreadPool
from crowd_detection import CrowdDetection

pool = ThreadPool(processes=1)
root = os.getcwd()

class VideoCapture:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class PlayVideo:
    def __init__(self, source, window_name, q):
        self.cap_line = None
        self.model = YOLO("yolov8s.pt")
        self.detections = None
        self.image = None
        self.source = source
        self.window_name = window_name
        self.q_img = q
        self.line_coord = None
        self.first_frame_processed = False

    def process_first_frame(self, image):
        clone = image.copy()
        cv2.imshow(f"Draw Line - {self.window_name}", clone)
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                cv2.circle(clone, (x, y), 4, (0, 0, 255), thickness=-1)
                cv2.imshow(f"Draw Line - {self.window_name}", clone)

        cv2.setMouseCallback(f"Draw Line - {self.window_name}", click_event)

        while True:
            if cv2.waitKey(1) & 0xFF == 13:  # 'Enter' key
                break

        cv2.destroyWindow(f"Draw Line - {self.window_name}")

        if len(points) >= 2:
            if points[0] != points[-1]:
                points.append(points[0])
            self.line_coord = np.array(points, dtype=np.int32)
            print(f"Line coordinates saved for {self.window_name}: {self.line_coord}")
        else:
            print("Not enough points to form a line.")

    def vdo_cap(self):
        try:
            if self.source.startswith("rtsp"):
                self.cap_line = VideoCapture(self.source)
            else:
                self.cap_line = cv2.VideoCapture(self.source)

            while True:
                if self.source.endswith((".mp4", ".avi")):
                    ret, self.image = self.cap_line.read()
                else:
                    self.image = self.cap_line.read()
                self.image = cv2.resize(self.image, (1080, 720))
                if not self.first_frame_processed:
                    self.process_first_frame(self.image)
                    self.first_frame_processed = True

                if self.line_coord is not None:
                    cv2.polylines(self.image, [self.line_coord], True, (0, 220, 0), 4)

                self.q_img.put(self.image)

                if self.line_coord is not None:
                    frame = pool.apply_async(CrowdDetection(self.model, self.line_coord, self.line_coord).predict,
                                             (self.q_img,)).get()
                else:
                    frame = self.image

                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    urls = [
        {"name": "Camera1", "url": r"rtsp://admin:Admin123$@10.11.25.64:554/stream1"},
        {"name": "Camera2", "url": r"rtsp://admin:Admin123$@10.11.25.65:554/stream1"},
    ]
    queue_list = []
    threads = []

    for i in urls:
        url = i['url']
        name = i["name"]
        q = queue.Queue()
        queue_list.append(q)
        td = threading.Thread(
            target=PlayVideo(url, name, q).vdo_cap)
        td.start()
        threads.append(td)

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
