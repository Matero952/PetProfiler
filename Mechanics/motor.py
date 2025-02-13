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
pwm = GPIO.PWM(ena, 22)
pwm.start(15)
#15 and 15 worked i think
#15 and 20 works

while True:
    GPIO.output(input1, GPIO.HIGH)
    GPIO.output(input2, GPIO.LOW)



#Reverse