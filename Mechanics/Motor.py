import RPi.GPIO as GPIO
import time as time
class Motor:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.input1 = 17
        self.input2 = 27
        self.ena = 4
        GPIO.setup(self.input1, GPIO.OUT)
        GPIO.setup(self.input2, GPIO.OUT)
        GPIO.setup(self.ena, GPIO.OUT)
        self.pwm = GPIO.PWM(self.ena, 5000)
    def open(self) -> None:
        try:
            limit = 0.43
            start_time = time.time()
            print(f"Start time: {start_time}")
            self.pwm.start(100)
            GPIO.output(self.input1, GPIO.HIGH)
            GPIO.output(self.input2, GPIO.LOW)
            while time.time() - start_time < limit:
                pass
            else:
                self.pwm.ChangeDutyCycle(5)
                print(f"Time limit reached.")
        except KeyboardInterrupt:
                GPIO.cleanup()
                print(f"Stopping motor...")
        return None
    def close(self) -> None:
        limit = 3
        start = time.time()
        GPIO.output(self.input1, GPIO.LOW)
        GPIO.output(self.input2, GPIO.HIGH)
        self.pwm.start(0)
        while True:
            try:
                user_input = input("Enter value:")
                if user_input != "q":
                    self.pwm.ChangeDutyCycle(0)
                else:
                    self.pwm.ChangeDutyCycle(45)
                    while time.time() - start < limit:
                        pass
                    else:
                        self.pwm.ChangeDutyCycle(0)
                        break
            except KeyboardInterrupt:
                GPIO.cleanup()
                print(f"Stopping motor...")
        return None
if __name__ == "__main__":
    motor = Motor()
    motor.open()
    print(f"Method finished.")
    time.sleep(5)
    motor.close()
