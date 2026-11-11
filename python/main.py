from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

CLOSED_ANGLE = 170
OPEN_ANGLE = 90

ui = WebUI()
angle = CLOSED_ANGLE


def move(new_angle):
    global angle
    angle = new_angle
    print(f"servo {angle}", flush=True)
    Bridge.call("servo", angle)
    ui.send_message("angle", {"angle": angle})


ui.on_message("open", lambda sid, data: move(OPEN_ANGLE))
ui.on_message("reset", lambda sid, data: move(CLOSED_ANGLE))
ui.on_connect(lambda sid: ui.send_message("angle", {"angle": angle}, room=sid))

move(CLOSED_ANGLE)
App.run()
