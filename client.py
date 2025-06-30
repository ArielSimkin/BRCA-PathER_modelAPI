import sys
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QMessageBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal, QObject

import zipfile
import socket
import ssl #encryption algorithem

class PipelineWorker(QObject):
    finished = pyqtSignal(str, str)  # emit extract_to and the used zip path to the next function after the worker finished to do its comunication with the server

    def __init__(self, slide_path, output_dir, client):
        super().__init__()
        self.zip_path = ""
        self.slide_path = slide_path
        self.output_dir = output_dir
        self.client = client

    def run(self):
        
        self.zip_path = self.create_zip()
        # Send the zipped slide to server
        self.client.send_clients_slide(self.zip_path, self.slide_path)

        # Count existing client_package folders and create a new one
        count = 1
        for item in os.listdir(self.output_dir):
            if os.path.isdir(os.path.join(self.output_dir, item)) and item.startswith("client_package"):
                count += 1
        folder_name = f"client_package{count+1}"
        extract_to = os.path.join(self.output_dir, folder_name)
        os.makedirs(extract_to, exist_ok=True)

        self.show_message("waiting", "wait till server is ready to get your slide")

        # Receive zipped result from server into extract_to folder
        zip_save_path = os.path.join(extract_to, "information.zip")
        self.client.rcive_information(zip_save_path)
        # Extract zip contents
        with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        self.client.client_socket.close()

        self.finished.emit(extract_to,  zip_save_path)

    def create_zip(self):
        zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{os.path.basename(self.slide_path)}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
            zipf.write(self.slide_path, arcname=os.path.basename(self.slide_path))
        return zip_path
    
    def show_message(self, title, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

class DZIViewer(QMainWindow):
    def __init__(self, html_paths):
        super().__init__()
        self.cls = ""
        self.html_paths = html_paths
        self.is_drawing = False

        self.setWindowTitle("BRCA-PathER")
        self.setGeometry(100, 100, 1200, 800)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        
        self.build_starting_window()

    def build_starting_window(self):
        front_layout = QVBoxLayout()
        front_layout.setAlignment(Qt.AlignCenter)
        front_layout.setSpacing(10)

        self.title = QLabel(f"welcome to\nBRCA-PathER")
        self.title.setStyleSheet("font-size: 40px; color: black; padding: 10px;")
        front_layout.addWidget(self.title)

        self.startButton = QPushButton("Import a slide")
        self.startButton.setFixedSize(300, 40)
        front_layout.addWidget(self.startButton)
        self.startButton.clicked.connect(self.import_button_got_clicked)

        self.reloadButtonFirst = QPushButton("reload an existing one")
        self.reloadButtonFirst.setFixedSize(300, 40)
        front_layout.addWidget(self.reloadButtonFirst)
        self.reloadButtonFirst.clicked.connect(self.reload_slides)

        self.main_layout.addLayout(front_layout)
        self.main_widget.setLayout(self.main_layout)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            elif item.layout() is not None:
                self.clear_layout(item.layout())

    def build_main_window(self, unzipped_folder):
        viewer_layout = QVBoxLayout()
        panel_layout = QVBoxLayout()

        self.org_viewer = QWebEngineView()
        viewer_layout.addWidget(self.org_viewer)
        self.maped_viewer = QWebEngineView()
        viewer_layout.addWidget(self.maped_viewer)

        org_html_path = os.path.abspath(os.path.join(unzipped_folder, self.html_paths[0]))
        mapped_html_path = os.path.abspath(os.path.join(unzipped_folder, self.html_paths[1]))

        self.org_viewer.load(QUrl.fromLocalFile(org_html_path))
        self.maped_viewer.load(QUrl.fromLocalFile(mapped_html_path))

        self.class_label = QLabel(f"Model's observation: {self.cls}")
        self.class_label.setStyleSheet("font-size: 16px; color: black; padding: 10px;")
        panel_layout.addWidget(self.class_label)

        self.maped_compare_button = QPushButton("Compare area with original image")
        self.original_compare_button = QPushButton("Compare area with maped image")
        self.toggle_draw_button = QPushButton("Toggle Drawing Mode")
        panel_layout.addWidget(self.maped_compare_button)
        self.maped_compare_button.clicked.connect(self.compare_mode)
        panel_layout.addWidget(self.original_compare_button)
        self.original_compare_button.clicked.connect(self.compare_mode)
        panel_layout.addWidget(self.toggle_draw_button)
        self.toggle_draw_button.clicked.connect(self.toggle_drawing_mode)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Write notes here...")
        panel_layout.addWidget(self.text_input)

        self.save_note = QPushButton("Save Notes")
        self.reload_button = QPushButton("Reload saved slides")
        self.import_button2 = QPushButton("Import another ")
        panel_layout.addWidget(self.save_note)
        panel_layout.addWidget(self.reload_button)
        panel_layout.addWidget(self.import_button2)
        self.save_note.clicked.connect(self.save_notes)
        self.import_button2.clicked.connect(self.import_button_got_clicked)
        self.reload_button.clicked.connect(self.reload_slides)

        self.main_layout.addLayout(viewer_layout, 4)
        self.main_layout.addLayout(panel_layout, 1)
        self.main_widget.setLayout(self.main_layout)

    def reload_slides(self):
        needed_files_or_folder = ["class.txt",  
                                  "maped.html",  
                                  "origin.html",  
                                  "original_image.dzi",  
                                  "original_image_files",  
                                  "slide.dzi",  
                                  "slide_files"]
        client_package = self.go_to_file_explorer(2)
        if client_package is None:
            return
        files = os.listdir(client_package)
        for file in needed_files_or_folder:
            if file not in files:
                return
        #get class
        with open(os.path.join(client_package, "class.txt"), "r") as f:
            self.cls = f.read().strip()
        self.clear_layout(self.main_layout)
        self.build_main_window(client_package)

    def main_process_manager(self, mode, path=""):
        if mode == 1:
            path = self.go_to_file_explorer(1)
            if path is None:
                return
        output_dir = self.go_to_file_explorer(2)
        if output_dir is None:
            return

        self.clear_layout(self.main_layout)

        self.waiting_screen()

        self.client = client_manegment("127.0.0.1", 8001)
        self.thread = QThread()
        self.worker = PipelineWorker(path, output_dir, self.client) #creating the synchronized network with the server for sending and getting the results

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_pipeline_finished) #function to run after the sytem finished with giving the resaults to the client
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_pipeline_finished(self, extract_to, zip_save_path):
        # Find file/folder that starts with "client_info"
        for name in os.listdir(extract_to):
            if name.startswith("client_info"):
                extract_to = os.path.join(extract_to, name)
                break

        #get class
        with open(os.path.join(extract_to, "class.txt"), "r") as f:
            self.cls = f.read().strip()

        #self.delete_zip_from_memory(zip_save_path)
        self.clear_layout(self.main_layout)
        self.build_main_window(extract_to)

    def go_to_file_explorer(self, mode):
        if mode == 1:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select WSI File", "", "Slide Files (*.svs)")
        elif mode == 2:
            file_path = QFileDialog.getExistingDirectory(self, "Select Folder for saving/reloading the information", ""
                                                         , QFileDialog.ShowDirsOnly)
        return file_path.replace("C:/", "/mnt/c/").replace("\\", "/") if file_path else None

    def import_button_got_clicked(self):
        self.main_process_manager(1)

    def waiting_screen(self):
        self.waiting_massage = QLabel("Waiting for model to finish...")
        self.waiting_massage.setStyleSheet("font-size: 16px; color: black; padding: 10px;")
        self.main_layout.addWidget(self.waiting_massage)

    def compare_mode(self):
        button = self.sender()
        if button.text() == "Compare area with original image":
            viewer1, viewer2 = self.maped_viewer, self.org_viewer
        else:
            viewer1, viewer2 = self.org_viewer, self.maped_viewer
        viewer1.page().runJavaScript("getZoomLevel();", lambda zoom:
            viewer1.page().runJavaScript("getCenter();", lambda center:
                self.sync_viewer(zoom, center, viewer2)))

    def sync_viewer(self, zoom, center, viewer):
        js = f"syncTo({zoom}, {center['x']}, {center['y']});"
        viewer.page().runJavaScript(js)

    def toggle_drawing_mode(self):
        for viewer in [self.org_viewer, self.maped_viewer]:
            viewer.page().runJavaScript("toggleDrawingMode();")

    def save_notes(self):
        text = self.text_input.toPlainText()
        path = self.go_to_file_explorer(2)
        if path:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "notes.txt"), "w", encoding="utf-8") as f:
                f.write(text)
    
    def delete_zip_from_memory(self, zip_path):
        os.remove(zip_path)

class client_manegment():
    def __init__(self, host, port):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_cert_chain(certfile="client.crt", keyfile="client.key")
        context.load_verify_locations("ca.pem")

        # IMPORTANT: set server_hostname to match what's in the certificate (e.g., 'localhost' or '127.0.0.1')
        raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket = context.wrap_socket(raw_socket, server_hostname="localhost")

        try:
            self.show_message("waiting", "waiting to connect to server")
            self.client_socket.connect(("localhost", port))
            self.show_message("connected" ,f"Connected to server at {host}:{port}")
        except Exception as e:
            print(f"Error occurred: {e}")

    def send_clients_slide(self, zip_path, path):
        try:
            slide_name = os.path.basename(path)
            name_bytes = slide_name.encode()
            
            file_size = os.path.getsize(zip_path)
            self.client_socket.sendall(len(name_bytes).to_bytes(4, 'big'))  # Name length
            self.client_socket.sendall(name_bytes)                          # Name
            self.client_socket.sendall(file_size.to_bytes(8, 'big'))        # File size

            with open(zip_path, "rb") as f:
                while True:
                    data = f.read(4096)
                    if not data:
                        break
                    self.client_socket.sendall(data)
        except Exception as e:
            print("something went wrong in server")

    def rcive_information(self, given_place):

        try:
            
            # Receive file size (8 bytes)
            file_size_data = self.client_socket.recv(8)
            file_size = int.from_bytes(file_size_data, 'big')

            with open(given_place, 'wb') as f:
                bytes_received = 0
                while bytes_received < file_size:
                    chunk = self.client_socket.recv(min(4096, file_size - bytes_received))
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_received += len(chunk)
        
        except Exception as e:
            print("something went wrong in server")

    def show_message(self, title, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DZIViewer(("origin.html", "maped.html"))
    viewer.show()
    sys.exit(app.exec_())
