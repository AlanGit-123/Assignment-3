#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
img = jetson.utils.loadImage("/home/nvidia/Desktop/train.jpg")     # '/dev/video0' for V4L2

detections = net.Detect(img)

for detection in detections:

	class_id = detection.ClassID
	confidence = detection.Confidence
	left = detection.Left
	top = detection.Top
	right = detection.Right
	bottom = detection.Bottom
	width = detection.Width
	height = detection.Height
	area = width * height
	centre_x = detection.Center[0]
	centre_y = detection.Center[1]

print(f"--ClassID: {class_id}")
print(f"--Confidence: {confidence:.2f}")
print(f"--Left: {left}")
print(f"--Top: {top}")
print(f"--Right: {right}")
print(f"--Bottom: {bottom}")
print(f"--Width: {width}")
print(f"--Height: {height}")
print(f"--Area: {area}")
print(f"--Centre: ({centre_x}, {centre_y})")

jetson.utils.saveImage("output_train.jpg", img)

	#print(detections)
	#display.Render(img)
	#display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

