import RPi.GPIO as GPIO

GPIO.setwarnings(False)
#775 motor
input1 = 17
input2 = 27
ena = 4
#Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(input1, GPIO.OUT)
GPIO.setup(input2, GPIO.OUT)
GPIO.setup(ena, GPIO.OUT)
#Normal
pwm = GPIO.PWM(ena, 100)
pwm.start(20)

try:
    while True:
        user_input = input()
        if user_input == "w":
            GPIO.output(input1, GPIO.HIGH)
            GPIO.output(input2, GPIO.LOW)
        elif user_input == "s":
            GPIO.output(input1, GPIO.LOW)
            GPIO.output(input2, GPIO.HIGH)
        elif user_input == "e":
            GPIO.output(input1, GPIO.LOW)
            GPIO.output(input2, GPIO.LOW)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()









#Reverse