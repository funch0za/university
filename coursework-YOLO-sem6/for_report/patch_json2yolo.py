"""Патчинг 274 строки general_json2yolo.py"""

with open('JSON2YOLO/general_json2yolo.py', 'r') as f:
    content = f.read()

old = 'h, w, f = img["height"], img["width"], img["file_name"]'
new = 'h, w, f = img["height"], img["width"], img["file_name"].split("/")[1]'

if old in content and new not in content:
    content = content.replace(old, new)
    with open('JSON2YOLO/general_json2yolo.py', 'w') as f:
        f.write(content)
