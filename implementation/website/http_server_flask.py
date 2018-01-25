import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from scipy.ndimage import imread
from scipy.misc import imsave

from flask import Flask, redirect, request, url_for
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        f = request.files["file"]
        I = imread(f.stream)
        print(I[:3,:3,:])
    return redirect(url_for("static", filename="index.html"))

