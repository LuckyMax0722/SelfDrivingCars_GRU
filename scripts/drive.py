import socketio
import eventlet.wsgi
import base64
import torch

from io import BytesIO
from PIL import Image
from flask import Flask
from lib.config import CONF
from lib.utils import jpg_to_tensor
from model.E2ERNN import E2ERNN

sio = socketio.Server()
application = Flask(__name__)


@sio.on('telemetry')
def telemetry(sid, data):
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))

    image_tensor = jpg_to_tensor(image)

    global image_sequence_tensor
    image_sequence_tensor = image_sequence_process(image_sequence_tensor, image_tensor)

    image_sequence_tensor_unsqueeze = image_sequence_tensor.unsqueeze(0)

    try:
        steering_angle = float(model(image_sequence_tensor_unsqueeze))  # predict the steering angel based on input image

        throttle = 0.1
        speed = 15

        send_control(steering_angle, throttle, speed)

    except Exception as e:
        print(e)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0.0, 0.0, 5)  # init


def send_control(steering_angle, throttle, speed):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),
        'speed': speed.__str__()
    }, skip_sid=True)


def image_sequence_process(image_sequence_tensor, image_tensor):
    for i in range(CONF.data.sequence_length - 1):
        image_sequence_tensor[i] = image_sequence_tensor[i + 1]
    image_sequence_tensor[-1] = image_tensor

    return image_sequence_tensor


if __name__ == '__main__':
    # load model
    model = E2ERNN()
    model.load_state_dict(torch.load(CONF.model.best_model))
    model.eval()

    # Image Sequence
    image_sequence_tensor = torch.zeros(CONF.data.sequence_length, 3, 160, 320)

    # connect to simulator
    application = socketio.Middleware(sio, application)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), application)
