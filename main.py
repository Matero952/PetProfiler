import cv2
import torch
class Pipeline:
    def __init__(self, motor, model, Capture, model_path):
        self.motor = motor
        self.model = model
        self.model_path = model_path
        self.Capture = Capture
    def pipeline(self) -> None:
        cap = cv2.VideoCapture('/dev/video0')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = 150
        count = 0
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                count += 1
                print(f"count: {count}")
                if count % interval == 0:
                    img = f"frames{count}.jpg"
                    cv2.imwrite(img, frame)
                    result = self.Capture.analyze(img)
                    print(f"Result: {result}")
                    if result == 1:
                        self.motor.open()
                        time.sleep(10)
                        self.motor.close()
                    else:
                        pass
                else:
                    pass
            except KeyboardInterrupt or KeyError:
                cap.release()
                cv2.destroyAllWindows()
                break
if __name__ == '__main__':
    from Mechanics import Motor
    from Vision.inspect import Capture
    import time
    from Network.CNN.model import CNN
    print(f"Script started (:")
    motor = motor.Motor()
    model = CNN(1, 1).to(device=torch.device("cpu"))
    model_path = "results/megusta/model_7epochs_lr:0.00052.pth"
    capture = Capture(model=model,  model_path=model_path)
    petprofiler = Pipeline(motor=motor, model=model, Capture=capture, model_path=model_path)
    while True:
        counter = 0
        petprofiler.pipeline()
        print("counter:", counter)
        counter += 1
