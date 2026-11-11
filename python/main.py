import time

from arduino.app_utils import App, Bridge

# Servo test: step through a few angles, then repeat
ANGLES = [0, 90, 180, 90]


def loop():
    for angle in ANGLES:
        print(f"servo {angle}")
        Bridge.call("servo", angle)
        time.sleep(2)


App.run(user_loop=loop)
