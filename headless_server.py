from datetime import date, datetime
from flask import Flask, request, send_file, make_response, redirect
import io
from PIL import Image
from modules.img2img import img2img
import threading

from modules.txt2img import txt2img
from modules.db_logger import getQueries
from urllib import request as rq
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
    im = None
    with rq.urlopen(postData)  as rsp:
        im = rsp.read()
    image = Image.open(io.BytesIO(im)).convert('RGB')
    return image

def create_img(w, h, color):
    image = Image.new('RGB', (w, h), color)
    return image

def hex_color_string_to_tuple(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

@app.route('/', methods=['GET'])
def index():
    return redirect(DEFAULT_UI_URL, code=302)

model_alias = {'compvis': 'sd-1.4-model'
              ,'waifu': 'wd-v1-3-float32'}

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
        if mod in model_alias.keys():
            ckpt = get_closet_checkpoint_match(model_alias[mod])
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
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        return response

    s = request.args.get("s")
    response = None
    if s is None or s == "":
        s = " "
    
    steps = request.args.get("steps")
    if steps is None or steps == "":
        steps = 20
        
    tiling = request.args.get("tiling")
    if tiling is None or tiling == "":
        tiling = False

    n_iter = request.args.get("n")
    if n_iter is None or n_iter == "":
        n_iter = 1
        
    height = request.args.get("h")
    if height is None or height == "":
        height = 512

    width = request.args.get("w")
    if width is None or width == "":
        width = 512
        
    denoising_strength = request.args.get("strength")
    if denoising_strength is None or denoising_strength == "":
        denoising_strength = 0.75
    
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
        init_image = load_img(postData.decode("utf-8"))
    
    args = (0, False, None, '', False, 1, '', 4, '', True, False)
    data = img2img(
        mode=0,
        prompt=s,
        negative_prompt='',
        prompt_style='',
        prompt_style2='',
        init_image=init_image,
        init_image_with_mask=None,
        init_img_inpaint=None,
        init_mask_inpaint=None,
        mask_mode=0,
        steps=steps,
        sampler_index=0,
        mask_blur=4,
        inpainting_fill=0,
        restore_faces=False,
        tiling=tiling,
        n_iter=n_iter,
        batch_size=1,
        cfg_scale=7.5,
        denoising_strength=denoising_strength,
        seed=0,
        subseed=0,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        height=height,
        width=width,
        resize_mode=0,
        inpaint_full_res=False,
        inpaint_full_res_padding=0,
        img2img_batch_input_dir='',
        img2img_batch_output_dir='',
        *args=*args)
    img_byte_arr = io.BytesIO()
    data[0][0].save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    response = make_response(mkResponse(img_byte_arr))
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/mod_mass', methods=['GET', 'POST', 'OPTIONS'])
def mod_mass():
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        return response

    s = request.args.get("s")
    response = None
    if s is None or s == "":
        s = " "
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
        init_image = load_img(postData.decode("utf-8"))
    
    count = request.args.get('count')
    if count is None or count == "":
        count = 1
    else:
        count = int(count)
    
    folderName = datetime.now().strftime("%Y%m%d%H%M%S") + "_temp"
    os.mkdir(folderName)
    args = (0, False, None, '', False, 1, '', 4, '', True, False)
    i = 0
    while i < count:
        data = img2img(0, s, '', '', '', init_image, None, None, None, 0, 20, 0, 4, 0, False, False, 1, 1, 7, 0.75, -1.0, -1.0, 0, 0, 0, False, 512, 512, 0, False, 32, 0, '', '', *args)
        data[0][0].save(f"{folderName}/image{i}.png", format='PNG')
        i += 1
    
    zipFileName = datetime.now().strftime("%Y%m%d%H%M%S") + 'images'
    shutil.make_archive(zipFileName, 'zip', folderName)
    for root, dirs, files in os.walk(folderName, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(folderName)

    data = []
    with open(zipFileName + '.zip', 'rb') as f:
        data = io.BytesIO(f.read())
    
    os.remove(zipFileName + '.zip')
    response = make_response(mkZipResponse(data))
    response.headers["Access-Control-Allow-Origin"] = "*"
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
