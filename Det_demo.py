import torch as th
import torchvision.transforms as T
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class DETRModel(object):
    # COCO classes
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

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    BBOX_COLOR = "red"
    BBOX_WIDTH = 2
    LABEL_FILL_COLOR = "red"

    def __init__(self, imageurl):
        self.imageurl = imageurl
        self.WIDTH, self.HEIGHT = 800, 600
        self.model, self.img, self.img_t, self.output = [None]*4
        self.init()

    def init(self):
        self.model_init()
        self.pre_process()

    def model_init(self):
        self.model = th.hub.load("facebookresearch/detr", 'detr_resnet50', pretrained=True).eval()

    def transform(self):
        t = T.Compose([
            # T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_t = t(self.img)
        self.img_t = self.img_t.unsqueeze(0)

    def download_image(self):
        self.img = Image.open(
            requests.get(self.imageurl, stream=True).raw
        ).resize((self.WIDTH, self.HEIGHT)).convert("RGB")

    def pre_process(self):
        self.download_image()
        self.transform()

    @staticmethod
    def get_font():
        custom_font = "/usr/local/lib/python3.6/dist-packages/werkzeug/debug/shared/ubuntu.ttf"
        font = ImageFont.truetype(custom_font, 20)
        return font

    @staticmethod
    def show_image(image):
        try:
            display(image)
        except:
            plt.imshow(image)

    def show_model_output(self):
        im = self.img.copy()
        drw = ImageDraw.Draw(im)
        for logit, box in zip(self.output['pred_logits'][0], self.output['pred_boxes'][0]):
            cls = logit.argmax()
            if cls >= len(self.CLASSES):
                continue
            label = self.CLASSES[cls]

            box = box * th.Tensor([800, 600, 800, 600])
            x, y, w, h = box
            x0, x1 = x - w // 2, x + w // 2
            y0, y1 = y - h // 2, y + h // 2

            drw.rectangle([x0, y0, x1, y1], width=self.BBOX_WIDTH, outline=self.BBOX_COLOR)
            drw.text((x, y), label, fill=self.LABEL_FILL_COLOR, font=self.get_font())
        self.show_image(im)

    def detect(self):
        with th.no_grad():
            self.output = self.model(self.img_t)
            self.show_model_output()


if __name__ == "__main__":

    # imageurl = "https://5.imimg.com/data5/GM/EM/MY-38731446/selection_143-500x500.png"
    imageurl = "https://www.siliconvalley.com/wp-content/uploads/2017/09/20160822__sjm-skully-0822-11-1.jpg?w=645"
    
    model = DETRModel(imageurl)
    model.detect()