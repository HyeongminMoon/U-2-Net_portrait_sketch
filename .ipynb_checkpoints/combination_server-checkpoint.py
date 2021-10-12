#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import base64
import cv2
import numpy as np
import io
import json
from flask import Flask, jsonify, request
from PIL import Image


# In[ ]:


def sketch_thick(img_path):
    target_url = "http://localhost:8895/sketch_thick"

    dict = {}
    dict['path'] = img_path

    response = requests.post(target_url, data=json.dumps(dict))

    data = json.loads(response.text)
    src = data['output_path']

    img = cv2.imread(src)

    return img, src


# In[ ]:


def img_to_imgpath(img):
    path = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/test_post.png'
    cv2.imwrite(path, img)
    
    return path


# In[ ]:


def cartoonize(img):
    target_url = "http://localhost:8894/predict"

    _, img_encoded = cv2.imencode('.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, 50])
    img = cv2.imdecode(img_encoded, 1)
    # send http request with image and receive response
    jpg_as_text = base64.b64encode(img_encoded).decode()
    dict = {}
    dict['image'] = jpg_as_text

    response = requests.post(target_url, data=json.dumps(dict))

    lists = json.loads(response.text)
    jpg_original = base64.b64decode(lists[0]['image'])
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    path = img_to_imgpath(img)
    
    return img, path


# In[ ]:


def remove_bg(img_path):
    target_url = "http://localhost:8895/remove_bg"

    dict = {}
    dict['img_path'] = img_path

    response = requests.post(target_url, data=json.dumps(dict))

    data = json.loads(response.text)
#     print(data)

    img = cv2.imread(data)
    return img, data


# In[ ]:




app = Flask(__name__)

    
@app.route('/sketch', methods=['POST'])
def sketch():
    if request.method == 'POST':
        
        #load request
        r = request
        data_json = r.data
        data_dict = json.loads(data_json)

        file_path = data_dict['img_path']
        
        img = cv2.imread(file_path)
        path = file_path

        # resize
        h, w = img.shape[:2]
        r = 360 / float(h)
        dim = (int(w * r), 360)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        #cartoonize
        img1, path1 = cartoonize(resized_img)
        img1 = cv2.resize(img1, (w, h))
        cv2.imwrite(path1, img1)

        #sketch
        img2, path2 = sketch_thick(path1)
   
        #cartoonize
        img3, path3 = cartoonize(img2)
        
        #write image
#         out_path = '/home/ubuntu/workspace/kobaco/sketchy/U-2-Net/test_data/test_portrait_images/api_results/api_image.png'
#         cv2.imwrite(out_path, img3)
        
        #return path
        result_dict = {}
        result_dict['output_path'] = path3
        return jsonify(result_dict)

# img, path = remove_bg(path)

# img, path = cartoonize(img)

# img, path = sketch_thick(path)


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8933, debug=False)


# In[ ]:


get_ipython().system('jupyter nbconvert --to ')


# In[ ]:




