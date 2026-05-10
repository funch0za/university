import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QListWidget, QLabel, QScrollArea,
                             QListWidgetItem, QFrame)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont

colors = np.random.randint(0, 255, (155, 3)).tolist()

class ImageViewer(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.labels_path = self.folder_path / "labels"
        
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(self.folder_path.glob(ext))
        self.image_files = sorted(self.image_files)
        
        self.setWindowTitle(f"Image Viewer - {folder_path}")
        self.setGeometry(100, 100, 1200, 700)
        
        self.setup_ui()
        
        if self.image_files:
            self.load_image(self.image_files[0])
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        self.create_left_panel(main_layout)
        self.create_center_panel(main_layout)
        self.create_right_panel(main_layout)
    
    def create_left_panel(self, layout):
        left_widget = QWidget()
        left_widget.setFixedWidth(250)
        left_layout = QVBoxLayout(left_widget)
        
        list_label = QLabel("Images")
        list_label.setFont(QFont("Arial", 12, QFont.Bold))
        list_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(list_label)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_item_clicked)
        
        for img_path in self.image_files:
            item = QListWidgetItem(img_path.name)
            item.setData(Qt.UserRole, img_path)
            self.file_list.addItem(item)
        
        left_layout.addWidget(self.file_list)
        layout.addWidget(left_widget)
    
    def create_center_panel(self, layout):
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        
        center_layout.addWidget(scroll)
        layout.addWidget(center_widget, 1)
    
    def create_right_panel(self, layout):
        right_widget = QWidget()
        right_widget.setFixedWidth(320)
        right_layout = QVBoxLayout(right_widget)
        
        legend_label = QLabel("Legend")
        legend_label.setFont(QFont("Arial", 12, QFont.Bold))
        legend_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(legend_label)
        
        self.legend_scroll = QScrollArea()
        self.legend_widget = QWidget()
        self.legend_layout = QVBoxLayout(self.legend_widget)
        self.legend_layout.setAlignment(Qt.AlignTop)
        self.legend_widget.setLayout(self.legend_layout)
        
        self.legend_scroll.setWidget(self.legend_widget)
        self.legend_scroll.setWidgetResizable(True)
        
        right_layout.addWidget(self.legend_scroll)
        layout.addWidget(right_widget)
    
    def on_item_clicked(self, item):
        img_path = item.data(Qt.UserRole)
        self.load_image(img_path)
    
    def load_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return
        
        label_path = self.labels_path / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            h, w = img.shape[:2]
            classes = set()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x = float(parts[1]) * w
                    y = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    conf = float(parts[5]) if len(parts) >= 6 else None
                    
                    x1, y1 = int(x - bw/2), int(y - bh/2)
                    x2, y2 = int(x + bw/2), int(y + bh/2)
                    
                    color = colors[class_id % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"class {class_id}"
                    if conf:
                        label += f" ({conf:.2f})"
                    
                    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    classes.add(class_id)
            
            self.update_legend(classes)
        
        self.display_image(img)
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == img_path:
                self.file_list.setCurrentRow(i)
                break
    
    def update_legend(self, classes):
        for i in reversed(range(self.legend_layout.count())):
            widget = self.legend_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        legend_h = 30 * len(classes) + 20
        legend = np.ones((legend_h, 300, 3), dtype=np.uint8) * 240
        
        y = 15
        for class_id in sorted(classes):
            color = colors[class_id % len(colors)]
            cv2.rectangle(legend, (10, y), (30, y+15), color, -1)
            cv2.putText(legend, f"Class {class_id}", (40, y+13), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            y += 30
        
        legend_rgb = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        h, w, _ = legend_rgb.shape
        bytes_per_line = 3 * w
        qimage = QImage(legend_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        legend_widget = QLabel()
        legend_widget.setPixmap(pixmap)
        self.legend_layout.addWidget(legend_widget)
    
    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        bytes_per_line = 3 * w
        qimage = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        screen_size = self.image_label.size()
        pixmap = QPixmap.fromImage(qimage)
        
        if screen_size.width() > 0 and screen_size.height() > 0:
            scaled_pixmap = pixmap.scaled(screen_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            scaled_pixmap = pixmap
        
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not Path(folder_path).exists():
        print(f"Folder not found: {folder_path}")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    window = ImageViewer(folder_path)
    window.show()
    sys.exit(app.exec_())
