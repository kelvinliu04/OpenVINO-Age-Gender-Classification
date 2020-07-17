import os
import sys
import logging as log
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

class AgeGender:
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name = model_name
        self.device = device
        self.extensions = extensions

    def load_model(self):
        ## Get model_bin and model_xml
        model_bin = self.model_name + ".bin"
        model_xml = self.model_name + ".xml"
        plugin = IECore()
        network = IENetwork(model=model_xml, weights=model_bin)
        ## Add extension if any
        if self.extensions and "CPU" in self.device:                # Add a CPU extension, if applicable
            plugin.add_extension(self.extensions, self.device)
        ## (Additional) Check unsupported layer 
        supported_layers = plugin.query_network(network=network, device_name=self.device)
        unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) > 2:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        ## Load network
        self.exec_network = plugin.load_network(network, self.device)                                        
        self.input_blob = next(iter(network.inputs))
        self.output_blob = next(iter(network.outputs))
        self.n, self.c, self.h, self.w = network.inputs[self.input_blob].shape
        self.plugin = plugin
        self.network = network
        
        
    def predict(self, image):
        image = self.preprocess_input(image)
        self.exec_network.requests[0].infer({self.input_blob: image})
        age = self.exec_network.requests[0].outputs['age_conv3'][0]
        prob = self.exec_network.requests[0].outputs["prob"][0]
        age, gender = self.preprocess_output(age, prob) 
        return age, gender

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        img = cv2.dnn.blobFromImage(image, size=(self.w, self.h))
        return img

    def preprocess_output(self, age, prob):
        label = (0, 1)
        age = int(round(age[0][0][0] * 100))
        gender = label[np.argmax(prob)]
        return age, gender

