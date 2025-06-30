#imports

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.utils import load_img, img_to_array
import openslide
from PIL import Image
from keras.applications.xception import preprocess_input
import threading
from openslide.deepzoom import DeepZoomGenerator
from openslide import ImageSlide
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import pyvips
from concurrent.futures import ThreadPoolExecutor
import shutil
import zipfile
import queue

import html_generator

import socket
import ssl #encryption algorithem

class model_agent:
    def __init__(self, WSI, client_socket ,model_path='model_info/AXAHv1.keras', tile_size=299):
        self.client_socket = client_socket #the agent will know what is his server thread's socket for sending later the ziped information back to the client

        self.slide_path = WSI

        #do with this the same things did with the zips and information zip: create a larger folder, and save in 
        # it the package directory of this specific agent task based on how many already exist in the folder
        self.package_name = "clients_packages" # making directory for the information folder the system creates
        os.makedirs(self.package_name, exist_ok=True)  # Create the folder if it doesn't exist

        self.AXAHv1_model = tf.keras.models.load_model(model_path)
        self.AXAHv1_model.trainable = False
        self.tile_size = tile_size
        self.slide = openslide.OpenSlide(WSI)
        self.predictions = []
        
        # Determine the maximum dimensions needed for the canvas
        max_x, max_y = self.slide.dimensions

        # Create a blank canvas with a white background
        self.canvas = pyvips.Image.black(max_x, max_y, bands=3).new_from_image([255, 255, 255])


    def normalize_tile(self, tile):
        tile_array = np.array(tile).astype(np.float32) / 255.0
        return Image.fromarray((tile_array * 255).astype(np.uint8))

    def is_informative(self, tile, std_threshold=50, mean_threshold=200):
        tile_np = np.array(tile)
        gray = np.mean(tile_np, axis=2)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        return not (mean_intensity > mean_threshold or std_intensity < std_threshold)

    def slide_processing(self, wsi_path, epsilon, maping_epsilon):
        width, height = self.slide.dimensions

        for x in range(0, width, self.tile_size):
            for y in range(0, height, self.tile_size):
                if np.random.rand() > epsilon:
                    if x + self.tile_size <= width and y + self.tile_size <= height:
                        tile = self.slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
                        if self.is_informative(tile):
                            tile = self.normalize_tile(tile)
                            self.predict(tile)
                            if np.random.rand() > maping_epsilon:
                                tile = self.make_silency_map(tile)
                            self.insert_into_deepzoom_slide((x, y), tile)
                        
    def overlay_heatmap(self, original_tile, heatmap):
        heatmap_resized = cv2.resize(heatmap, (self.tile_size, self.tile_size))
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(np.uint8(original_tile), 0.6, heatmap_color, 0.4, 0)
        return Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))

    def predict(self, tile):
        tile_array = tile
        tile_array = np.expand_dims(tile_array, axis=0)
        tile_array = preprocess_input(tile_array)

        prediction = self.AXAHv1_model.predict(tile_array)
        cl = (prediction > 0.5).astype(int).flatten()
        self.predictions.append(cl[0])

    def get_img_array(self, img, size):
        array = img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return tf.keras.applications.xception.preprocess_input(array)

    # Create Grad-CAM heatmap
    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap, pred_index

    def make_silency_map(self, tile):

        # Prepare input image
        img_array = self.get_img_array(tile, size=(299, 299))

        # Generate heatmap
        heatmap, pred_index = self.make_gradcam_heatmap(img_array, self.AXAHv1_model, 'block14_sepconv2_act')

        # Resize heatmap to 299x299
        heatmap_resized = cv2.resize(heatmap, (299, 299))

        # Create color map
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

        # Superimpose heatmap on image
        superimposed = cv2.addWeighted(np.uint8(tile), 0.6, heatmap_color, 0.4, 0)

        return superimposed

    def insert_into_deepzoom_slide(self, pos, tile):
        # Convert the PIL image to a NumPy array
        tile_np = np.array(tile)
        # Create a pyvips image from the NumPy array
        tile_vips = pyvips.Image.new_from_array(tile_np)
        # Insert the tile into the canvas
        self.canvas = self.canvas.insert(tile_vips, pos[0], pos[1])

    def scan_wsi_and_generate_slide(self, wsi_path, epsilon, maping_epsilon, slide_output_dir):

        self.slide_processing(wsi_path, epsilon, maping_epsilon)

        # Save the assembled image as a Deep Zoom Image
        self.canvas.dzsave(slide_output_dir)

        pred_ratio_of_classes = self.predictions.count(0) / len(self.predictions)
    
        return 0 if pred_ratio_of_classes > 0.01 else 1
    
    def convert_svs_to_dzi(self, input_svs, output_prefix):
        # Load the .svs file
        image = pyvips.Image.new_from_file(input_svs, access='sequential')

        # Generate the DZI pyramid
        image.dzsave(output_prefix)
    
    def start(self):

        count = 1
        for file in os.listdir(self.package_name):
            if file.endswith("client_info"):
                count += 1

        # Name the next package with the next count number
        specific_package_filename = f"client_info{count}"
        
        package = os.path.join(self.package_name, specific_package_filename)
        
        os.makedirs(package, exist_ok=True)
        
        answer = self.scan_wsi_and_generate_slide(self.slide_path, epsilon = 0, maping_epsilon = 0, slide_output_dir= os.path.join(package, 'slide')) #saves in a packeg that will later be sended to the client
        
        self.convert_svs_to_dzi(self.slide_path, os.path.join(package, "original_image")) #saves in a packeg that will later be sended to the client

        status = self.send_all_information_to_client(answer, package)

        return status

    def send_all_information_to_client(self, answer, package):

        # saving the html files in the created folder
        html1 = html_generator.get_html_file(1)
        html2 = html_generator.get_html_file(2)

        with open(os.path.join(package, 'origin.html'), "w", encoding="utf-8") as f:
            f.write(html1)
        with open(os.path.join(package, 'maped.html'), "w", encoding="utf-8") as f:
            f.write(html2)

        #save predicted class
        answer = "ER+" if answer == 0 else "ER-"
        with open(os.path.join(package, 'class.txt'), "w", encoding="utf-8") as f:
            f.write(answer)
        
        #ziping the full folder for sending it to the client
        # Get parent directory of the folder you're zipping
        parent_dir = os.path.dirname(self.package_name)

        # Create a 'zips' folder if it doesn't exist
        zips_dir = os.path.join(parent_dir, "zips")
        os.makedirs(zips_dir, exist_ok=True)
        
        count = 1
        for file in os.listdir(zips_dir):
            if file.startswith("information") and file.endswith(".zip"):
                count += 1
        
        # Name the next zip with the next count number
        zip_filename = f"information{count}.zip"
        zip_path = os.path.join(zips_dir, zip_filename) 

        f_p = self.create_zip(zip_path)

        status = server.send_clients_information(zip_path, answer, self.client_socket)

        self.delete_from_memory(f_p)
        self.delete_from_memory(self.slide_path + ".zip")
        self.delete_from_memory(self.slide_path)

        return status

    def create_zip(self, zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.package_name):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.package_name)
                    zipf.write(full_path, arcname=rel_path)
                       
        return zip_path

    def delete_from_memory(self, zip_path):
        os.remove(zip_path)
        
class server_manegment():
    def __init__(self):
        self.context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.context.load_cert_chain(certfile="server.crt", keyfile="server.key")
        self.context.load_verify_locations("ca.pem")
        self.context.verify_mode = ssl.CERT_REQUIRED

        self.client_slide_zips_path = "clients_slides" #create directory with slide for saving the slide zips of clients
        os.makedirs(self.client_slide_zips_path, exist_ok=True)  # Create the folder if it doesn't exist

        self.host, self.port = "127.0.0.1", 8001

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f"[*] Server listening on {self.host}:{self.port}")

        self.clients_queue = queue.Queue()

    def process_queue(self):
        while True:
            ssl_client_socket, client_address = self.clients_queue.get()
            print(f"[>] Handling client {client_address} from queue")
            self.handle_client(ssl_client_socket, client_address)
            self.clients_queue.task_done()

    def setup_server(self):
        # Start the queue processing thread (one worker)
        threading.Thread(target=self.process_queue, daemon=True).start()

        try:
            while True:
                client_socket, client_address = self.server_socket.accept()
                ssl_client_socket = self.context.wrap_socket(client_socket, server_side=True)
                print("\n[+] server completed handshake")
                # Add the socket + address to queue
                self.clients_queue.put((ssl_client_socket, client_address))

        except KeyboardInterrupt:
            print("\n[!] Server shutting down.")
            self.server_socket.close()

    def handle_client(self, client_socket, client_address):
        print(f"[+] New connection from {client_address}")
        try:
            # Receive 4 bytes for length
            size_bytes = client_socket.recv(4)

            name_size = int.from_bytes(size_bytes, byteorder='big')

            path = client_socket.recv(name_size).decode()#name of the slide so the server will know what to open inside the eception folder

            slide_path = os.path.join(self.client_slide_zips_path, path)

            self.recive_clients_slide(slide_path, client_socket)

            #extract zip
            with zipfile.ZipFile(slide_path + ".zip", 'r') as zip_ref:
                zip_ref.extractall(self.client_slide_zips_path)

            agent = model_agent(slide_path, client_socket)
            agent.start()

            client_socket.close()

        except Exception as e:
            print(f"[{client_address}] Error: {e}")
        
    def recive_clients_slide(self, slide_path, client_socket):
        try:
            # Receive file size (8 bytes)
            file_size_data = client_socket.recv(8)
            file_size = int.from_bytes(file_size_data, 'big')

            with open(slide_path + ".zip", 'wb') as f:
                bytes_received = 0
                while bytes_received < file_size:
                    chunk = client_socket.recv(min(4096, file_size - bytes_received))
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_received += len(chunk)

        except Exception as e:
            print("something went wrong in connection")

    def send_clients_information(self, saved_zip_place, answer, client_socket):
        
        try:
            
            #send the zip that already has been made and saved in the directory saved_zip_place with a spaciel name that comes in this parameter already
            file_size = os.path.getsize(saved_zip_place)
            client_socket.sendall(file_size.to_bytes(8, 'big'))        # File size

            with open(saved_zip_place, "rb") as f:
                while True:
                    data = f.read(4096)
                    if not data:
                        break
                    client_socket.sendall(data)

            return True

        except Exception as e:
            print("something went wrong in connection")

if __name__ == "__main__":
    server = server_manegment()
    server.setup_server()
