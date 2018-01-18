#!/usr/bin/env python

import base64
import json

from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer
from scipy.ndimage import imread
from scipy.misc import imsave

from ..models.ResNet18 import pretrained_res18
from ..models.ModelWrapper import ModelWrapper
from ..utils import heat_map
from ..attackers.BlackBoxAttacker import PGDAttackBB, GANAttack, WhiteNoiseAttack
from ..attackers.WhiteBoxAttacker import PGDAttack


def parse_as_dict(part):
    """
    Example of input:
    Content-Disposition: form-data; name="file"; filename="test.txt"
    Content-Type: image/jpeg
    """
    parsed = {}
    value = b""
    started_reading_value = False
    for line in filter(len, part.split(b"\r\n")):
        if not started_reading_value and line.startswith(b"Content"):
            for entry in line.split(b"; "):
                try:
                    k, v = entry.split(b": ")
                except ValueError:
                    k, v = entry.split(b"=")
                    v = v.strip(b'"')
                parsed[k] = v
        else:
            started_reading_value = True
            value += line
            value += b"\r\n"
    parsed[b"value"] = value
    return parsed


def retrieve_part(post_data, name, boundary):
    parts = filter(len, post_data.split(boundary))
    for part in parts:
        parsed = parse_as_dict(part)
        if parsed.get(b"name") == name:
            return parsed
    return None


def cvt2b64(I, fmt="png"):
    with BytesIO() as fout:
        imsave(fout, I, fmt)
        b64 = base64.b64encode(fout.getbuffer())
    return b64.decode("ascii")
    

def benign_results(image, title):
    wrapper_base = ModelWrapper(model_base)
    wrapper_robust = ModelWrapper(model_robust)
    result = {}
    img_original, probdist1 = wrapper_base(image, return_img=True)
    probdist2 = wrapper_robust(image)
    result = {
        "title": title,
        "image1": cvt2b64(img_original),
        "caption1": "Original image",
        "pdf1": probdist1,
        "pdf2": probdist2,
    }
    return result, img_original


def attack_results(image, attack, title):
    img_pert, probdist1 = attack(model_base, image, return_img=True)
    probdist2 = attack(model_robust, image, return_img=False)
    hm = heat_map(img_pert, image)
    result = {
        "title": title,
        "image1": cvt2b64(img_pert),
        "image2": cvt2b64(hm),
        "caption1": "Perturbed image",
        "caption2": "Differences heat map",
        "pdf1": probdist1,
        "pdf2": probdist2,
    }
    print(probdist1)
    print(probdist2)
    return result


# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    
    def _set_headers(self, status, headers):
        # Send response status code
        self.send_response(status)
        # Send headers
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()

  # GET
    def do_GET(self):
        self._set_headers(200, {
            "Content-type": "text/plain",
        })
        # Send message back to client
        message = "Nothing to display..."
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return
        
    def do_POST(self):
        try:
            # Get posted data
            content_length = int(self.headers["Content-Length"]) # <--- Gets the size of data
            boundary = "--" + self.headers["Content-Type"].split("; ")[1].split("=")[1] # boundary is -- + this field's value
            post_data = self.rfile.read(content_length) # <--- Gets the data itself
            # Retrieve part corresponding to image field
            part = retrieve_part(post_data, b"file", boundary.encode("ascii"))
            image_b = part[b"value"]
            # Convert image to Numpy
            fin = BytesIO(image_b)
            with BytesIO(image_b) as fin:
                I = imread(fin)
            # Set headers and set output
            self._set_headers(200, {
                "Content-type": "application/json",
                "Access-Control-Allow-Origin": "*",
            })
            # write results
            print("Benign example...")
            benign, cropped = benign_results(I, "Benign example")
            print("Performing random noise attack...")
            attack1 = attack_results(cropped, rand_attack, "Random noise")
            print("Performing whitebox PGD...")
            attack2 = attack_results(cropped, pgd_attack, "Whitebox PGD")
            print("Performing blackbox PGD...")
            attack3 = attack_results(cropped, pgd_attack_bb, "Blackbox PGD")
            print("Performing blackbox GAN...")
            attack4 = attack_results(cropped, gan_bb, "Blackbox GAN")
            self.wfile.write(json.dumps([
                benign,
                attack1,
                attack2,
                attack3,
                attack4,
            ]).encode("ascii"))
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            self._set_headers(500, {
                "Content-type": "text/plain",
                "Access-Control-Allow-Origin": "*",
            })
            self.wfile.write(str(e).encode("ascii"))


def run():
    print('starting server...')
    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('127.0.0.1', 8081)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()


if __name__ == "__main__":
    model_base = pretrained_res18(which=0)
    model_robust = pretrained_res18(which=2)
    model_att = pretrained_res18(which=3)
    rand_attack = WhiteNoiseAttack(intensity=1.0)
    pgd_attack = PGDAttack(epsilon=0.05, a=0.01, k=5)
    pgd_attack_bb = PGDAttackBB(model_att, epsilon=0.05, a=0.01, k=5)
    gan_bb = GANAttack(None, intensity=0.2)
    run()
