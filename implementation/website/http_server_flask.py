import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import base64
import json

from io import BytesIO

from scipy.ndimage import imread
from scipy.misc import imsave

from implementation.utils import heat_map
from implementation.models.ResNet18 import pretrained_res18
from implementation.models.ModelWrapper import ModelWrapper
from implementation.attackers.BlackBoxAttacker import PGDAttackBB, GANAttack, WhiteNoiseAttack
from implementation.attackers.WhiteBoxAttacker import PGDAttack

from flask import Flask, redirect, request, url_for
app = Flask(__name__)


model_base = pretrained_res18(which=0)
model_robust = pretrained_res18(which=2)
model_att = pretrained_res18(which=3)
rand_attack = WhiteNoiseAttack(intensity=1.0)
pgd_attack = PGDAttack(epsilon=0.05, a=0.01, k=5)
pgd_attack_bb = PGDAttackBB(model_att, epsilon=0.05, a=0.01, k=5)
gan_bb = GANAttack(None, intensity=0.2)


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
    # print(probdist1)
    # print(probdist2)
    return result


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        f = request.files["file"]
        try:
            I = imread(f.stream)
            headers = {
                    "Content-type": "application/json",
                    "Access-Control-Allow-Origin": "*",
            }
            status = 200
            print("Doing benign example...")
            benign, cropped = benign_results(I, "Benign example")
            print("Performing random noise attack...")
            attack1 = attack_results(cropped, rand_attack, "Random noise")
            print("Performing whitebox PGD...")
            attack2 = attack_results(cropped, pgd_attack, "Whitebox PGD")
            print("Performing blackbox PGD...")
            attack3 = attack_results(cropped, pgd_attack_bb, "Blackbox PGD")
            print("Performing blackbox GAN...")
            attack4 = attack_results(cropped, gan_bb, "Blackbox GAN")
            content = json.dumps([benign, attack1, attack2, attack3, attack4])
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            content = str(e)
            headers = {
                    "Content-type": "text/plain",
            }
            status = 500
        return content, status, headers
    return redirect(url_for("static", filename="index.html"))

