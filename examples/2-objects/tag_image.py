import cv2
from bing_image_downloader import downloader
from typing import List
import os


class ImageDownloader:
    def __init__(self, q: str, length: int, out_dir: str) -> None:
        self.q = q
        self.length = length
        self.out_dir = out_dir

    def perform(self):
        downloader.download(
            self.q, 
            limit = self.length, 
            output_dir = self.out_dir,
            adult_filter_off = True, 
            force_replace = False, 
            timeout = 60, 
            verbose = True
        )


class Classifier:
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.current_rect: List[int] = []
        self.images = iter(os.listdir(dir))
        self.window_name = "classify"
        self.current_image = None
        print(self.images)

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('x = %d, y = %d'%(x, y))

            self.current_rect = self.current_rect + [x, y]
            if len(self.current_rect) == 4:
                x1,y1, x2, y2 = self.current_rect
                cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.imshow(self.window_name, self.current_image)
                print('drawrect', self.current_rect, event) 
                self.current_rect = []

    def next(self):
        image_filename = next(self.images)
        if not image_filename:
            return False 
    
        img_path = self.dir + "/" + image_filename
        self.current_image  = cv2.imread(img_path)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)
        cv2.imshow(self.window_name, self.current_image)

        key = cv2.waitKey()
        # 78 = n
        if key == 78:
            return True

        return True


images_dir = "/downloaded_images"
ImageDownloader("smartphone front", 100, images_dir).perform()

classify = Classifier(images_dir)


while classify.next():
    print("next")