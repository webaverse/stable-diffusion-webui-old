from datetime import date, datetime
from flask import Flask, request, send_file, make_response, redirect
import io
from PIL import Image
from modules.img2img import img2img
import threading

from modules.txt2img import txt2img
from modules.db_logger import getQueries
from modules.sd_models import reload_model_weights, checkpoint_tiles, get_closet_checkpoint_match
import modules.shared as shared

import shutil
import os

DEFAULT_UI_URL = "http://127.0.0.1:7860"
app = Flask(__name__)
jobLock = threading.Lock()

def mkResponse(data):
  return send_file(
    data,
    download_name="image.png",
    mimetype="image/png",
  )

def mkZipResponse(data):
    return send_file(
        data,
        download_name="images.zip",
        mimetype="application/zip",
    )

def load_img(postData):
    image = Image.open(io.BytesIO(postData)).convert('RGB')
    return image

def create_img(w, h, color):
    image = Image.new('RGB', (w, h), color)
    return image

def hex_color_string_to_tuple(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

@app.route('/', methods=['GET'])
def index():
    return redirect(DEFAULT_UI_URL, code=302)

@app.route('/image', methods=['GET', 'OPTIONS'])
def image():
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
    elif jobLock.acquire(timeout=10):

        s = request.args.get("s")
        mod = request.args.get("model")
        response = None
        if mod is not None:
            ckpt = get_closet_checkpoint_match(mod)
            if ckpt is None:
                response = make_response("no match for checkpoint \"{}\" found".format(mod), 400)
            elif ckpt != shared.opts.sd_model_checkpoint:
                shared.opts.sd_model_checkpoint = ckpt
                reload_model_weights(shared.sd_model)
        if s is None or s == "":
            response = make_response("no text provided", 400)
        if response is None:
            prompt = s
            args = (0, False, None, '', False, 1, '', 4, '', True, False)
            data = txt2img(prompt, '', '', '', 20, 0, 0, 0, 1, 1, 7, -1, -1, 0, 0, 0, False, 512, 512, False, False, 0.75, *args)
            img_byte_arr = io.BytesIO()
            data[0][0].save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            response = make_response(mkResponse(img_byte_arr))
            response.headers["Access-Control-Allow-Origin"] = "*"
        jobLock.release()
    else:
        response = make_response("Server busy", 409)
    return response

@app.route('/mod', methods=['GET', 'POST', 'OPTIONS'])
def mod():
    response = None
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
    elif jobLock.acquire(timeout=10):
        s = request.args.get("s")
        if mod is not None:
            ckpt = get_closet_checkpoint_match(mod)
            if ckpt is None:
                response = make_response("no match for checkpoint \"{}\" found".format(mod), 400)
            elif ckpt != shared.opts.sd_model_checkpoint:
                shared.opts.sd_model_checkpoint = ckpt
                reload_model_weights(shared.sd_model)
        if s is None or s == "":
            response = make_response("no text provided", 400)
        if response is None:
            prompt = s
            init_image = None
            if request.method == "GET":
                color_hex = None
                if request.args.get("color") is not None:
                    color_hex = request.args.get("color")
                else:
                    color_hex = "FFFFFF"
                color_tuple = hex_color_string_to_tuple(color_hex)
                init_image = create_img(512, 512, color_tuple)
            else:
                postData = request.get_data()
                init_image = load_img(postData)

            args = (0, False, None, '', False, 1, '', 4, '', True, False)
            data = img2img(0, prompt, '', '', '', init_image, None, None, None, 0, 20, 0, 4, 0, False, False, 1, 1, 7, 0.75, -1.0, -1.0, 0, 0, 0, False, 512, 512, 0, False, 32, 0, '', '', *args)
            img_byte_arr = io.BytesIO()
            data[0][0].save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            response = make_response(mkResponse(img_byte_arr))
            response.headers["Access-Control-Allow-Origin"] = "*"
        jobLock.release()
    else:
        response = make_response("Server busy", 409)
    return response

@app.route('/db_data', methods=['GET'])
def db_data():
    return getQueries()

@app.route('/list_ckpt', methods=['GET'])
def list_ckpt():
    return checkpoint_tiles() 

def run_server(host = '0.0.0.0', port=80):
    print("Starting server...")
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
    )

if __name__ == "__main__":
    run_server()
