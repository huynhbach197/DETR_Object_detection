import torch as th
import torchvision.transforms as T
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import sys
model = th.hub.load("facebookresearch/detr", 'detr_resnet50', pretrained=True).eval()
model = model.cuda()

transform = T.Compose([
            # T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
img_path = sys.argv[1]
img = Image.open(img_path).resize((800, 600)).convert('RGB')

img_tens = transform(img).unsqueeze(0).cuda()

with th.no_grad():
    output = model(img_tens)


im2 = img.copy()
drw = ImageDraw.Draw(im2)
pred_logits = output['pred_logits'][0]
pred_boxes = output['pred_boxes'][0]
max_output = pred_logits[:, :len(CLASSES)].softmax(-1).max(-1)
topk = max_output.values.topk(100)

pred_logits = pred_logits[topk.indices]
pred_boxes = pred_boxes[topk.indices]


for logits, box in zip(pred_logits, pred_boxes):
    cls = logits.argmax()
    if cls >= len(CLASSES):
        continue
    label = CLASSES[cls]
    print(label)
    box = box.cpu() * th.Tensor([800, 600, 800, 600])
    x, y, w, h = box
    x0, x1 = x - w // 2, x + w // 2
    y0, y1 = y - h // 2, y + h // 2

    drw.rectangle([x0, y0, x1, y1], width=1, outline='red')
    drw.text((x, y), label, fill='blue')
im2.show()