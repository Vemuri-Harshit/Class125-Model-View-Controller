import numpy as np;
import pandas as pd;
import PIL.ImageOps as IO;
from PIL import Image;
from sklearn.datasets import fetch_openml;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;

x,y = fetch_openml("mnist_784", version = 1, return_X_y = True);

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=7500, test_size=2500, random_state=5);
x_train_scale = x_train/255.0;
x_test_scale = x_test/255.0;

clf = LogisticRegression(solver='saga', multi_class = 'multinomial').fit(x_train_scale, y_train);

def getpredictions(image):
    im_pill = Image.open(image);
    im_bw = im_pill.convert('L');
    imbw_resized = im_bw.resize((28,28), Image.ANTIALIAS);
    pixel_filter = 20;
    min_pixel = np.percentile(imbw_resized, pixel_filter);
    imbw_resized_inverted = np.clip(imbw_resized - min_pixel, 0 ,255);
    max_pixel = np.max(imbw_resized)
    imbw_resized_inverted = np.asarray(imbw_resized_inverted) / max_pixel;
    test_sample = np.array(imbw_resized_inverted).reshape(1,784);
    test_pred = clf.predict(test_sample);
    return test_pred[0];

    