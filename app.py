from fileinput import filename
from flask import Flask, request
import base64
import json
import torch
import numpy as np
import cv2

import os

app = Flask(__name__)

# if loading custom model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path = "/".join([ os.getcwd(),'yolov5x.pt',]), force_reload = True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

@app.route("/postfile", methods=['GET', 'POST'])
def postFile():
    if request.method == 'POST':
        output_dict = {}
        my_string = ""
        content = request.get_json()
        image64 = bytes(content.get('image'), 'utf-8')
        file_name = "detect.jpg"

        with open(file_name, "wb") as fh:
            fh.write(base64.decodebytes(image64))
        
        result = model(file_name)
        result.save(".")

        with open(file_name, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())

        output_dict = json.loads(result.pandas().xyxy[0].to_json())
        output_dict['image-64'] = my_string.decode("utf-8") 

        return json.dumps(output_dict)
    else:
        return "- GET REQUEST -"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host='0.0.0.0',port=port)
