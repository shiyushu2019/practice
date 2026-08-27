import os
import xml.etree.ElementTree as ET
import shutil
import random

def convert_voc_to_yolo(xml_path, txt_path, class_mapping):
    """将 VOC XML 转换为 YOLO TXT"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class_mapping:
                continue
            class_id = class_mapping[name]
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

class_mapping = {
    'RBC': 0,
    'WBC': 1,
    'Platelets': 2
}

dataset_dir = './BCCD_Dataset'
images_dir = os.path.join(dataset_dir, 'JPEGImages')
annotations_dir = os.path.join(dataset_dir, 'Annotations')

# 检查数据集是否存在
if not os.path.exists(dataset_dir):
    print(f"❌ 错误: 找不到 {dataset_dir} 文件夹！")
    exit(1)

image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"找到 {len(image_files)} 张图片")

random.seed(42)
random.shuffle(image_files)

total = len(image_files)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")
print(f"测试集: {len(test_files)} 张")

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)

for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
    print(f"正在处理 {split} 集...")
    for img_file in files:
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(dataset_dir, split, 'images', img_file)
        shutil.copy2(src_img, dst_img)
        
        xml_file = img_file.replace('.jpg', '.xml').replace('.png', '.xml').replace('.jpeg', '.xml')
        src_xml = os.path.join(annotations_dir, xml_file)
        
        txt_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        dst_txt = os.path.join(dataset_dir, split, 'labels', txt_file)
        
        if os.path.exists(src_xml):
            convert_voc_to_yolo(src_xml, dst_txt, class_mapping)
        else:
            open(dst_txt, 'w').close()

print("✅ 数据集转换完成！")