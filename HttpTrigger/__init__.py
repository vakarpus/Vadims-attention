import logging

import azure.functions as func
import tensorflow as tf

from urllib.request import urlopen
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import time
from io import StringIO, BytesIO

heatmap_scema = [
    [0.0, (0, 0, 0)],
    [0.20, (0, 0, 0)],
    [0.30, (0, 0, 0.3)],
    [0.50, (0, 0.5, 0)],
    [0.70, (0, .7, 0.2)],
    [1.00, (0, 1.0, 0)],
]


def pixel(x, width=256, map=[], spread=1):
    width = float(width)
    r = sum([gaussian(x, p[1][0], p[0] * width, width/(spread*len(map))) for p in map])
    g = sum([gaussian(x, p[1][1], p[0] * width, width/(spread*len(map))) for p in map])
    b = sum([gaussian(x, p[1][2], p[0] * width, width/(spread*len(map))) for p in map])
    return min(1.0, r), min(1.0, g), min(1.0, b)


def gaussian(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def run(url):
    start_time = time.time()


    response = f"URL: {url} \r\n"
    K.clear_session()
    model = VGG16(weights=None) #weights='imagenet')
    response += f"Initializign weights: {time.time() - start_time :.2f} sec \r\n"

    model.load_weights("HttpTrigger/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    response += f"+Loading weights: {time.time() - start_time :.2f} sec \r\n \r\n"

    f = urlopen(url)
    img_org = Image.open(f)
    img = img_org.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    for i in range(3):
        res = decode_predictions(preds, top=3)[0][i]
        response += f"{res[1]} - {res[2]*100:.2f}%\r\n"

    ind = np.argmax(preds[0])

    vector = model.output[:, ind]

    # The output feature map of the `block5_conv3` layer, the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('block5_conv3')

    # The gradient of the vector class with regard to the output feature map of `block5_conv3`
    grads = K.gradients(vector, last_conv_layer.output)[0]

    # A vector of shape (512,), where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`, given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays, given the image
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array by "how important this channel is" with regard to the predicted class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    RGBheat = []
    for line in heatmap:
        RGBheat.append([])
        for x in line:
            r, g, b = pixel(x, width=1, map=heatmap_scema)
            r, g, b = [int(256*v) for v in (r, g, b)]
            pix = (r, g, b)
            RGBheat[-1].append(pix)

    heatmap = np.array(RGBheat)
    heatmap = np.uint8(heatmap)
    heatmap = np.expand_dims(heatmap, axis=0)
    
    sess = tf.Session()
    with sess.as_default():
        heatmap = tf.image.resize_images(heatmap, img_org.size[::-1], align_corners=True).eval()[0]
    heatmap = np.uint8(heatmap)


    
    superimposed_img = heatmap * 0.8 + img_org
    result_img = image.array_to_img(superimposed_img)
    
    draw = ImageDraw.Draw(result_img)
    font = ImageFont.load_default()
    
    response += f"\r\nTotal execution time: {time.time() - start_time :.2f} sec\r\n"
    
    draw.text( (10,10), response, (255, 255, 255), font=font)
    result_img.save('test.jpg')
    return result_img
    #return response    

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    url = req.params.get('url')
    if not url:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            url = req_body.get('url')

    if url:
        img = run(url)
        with BytesIO() as output:
            img.save(output, format="jpeg")
            return func.HttpResponse(output.getvalue(), mimetype="image/jpeg")
        return func.HttpResponse(img)
    else:
        return func.HttpResponse(
             "Please pass a name on the query string or in the request body",
             status_code=400
        )

url = "https://upload.wikimedia.org/wikipedia/commons/6/67/Dalmatiner_3.jpg"
img = run(url)
#with BytesIO() as output:
#    img.save(output, format="jpeg")
#    print(output.getvalue())
print(img)
