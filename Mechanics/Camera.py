import cv2
class Camera:
    def __init__(self, device_id):
        self.device_id = device_id

    def capture(self) -> None:
        cap = cv2.VideoCapture('/dev/video0')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                return None
            else:
                pass

    
