#####################################################################################
#
# MRGCV Unizar - Computer vision - Lib Test
#
# Title: Library test
#
# Date: 25 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################


import cv2
import numpy as np
import scipy as sc
import scipy.linalg as scAlg
import matplotlib.pyplot as plt
import time
import scipy.io as sio

from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots




sift = cv2.xfeatures2d.SIFT_create()

print('Hello')