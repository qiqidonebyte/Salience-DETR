import os

import albumentations as A
import cv2
import numpy as np
import torchvision

from transforms import v2 as T
from transforms.convert_coco_polys_to_mask import ConvertCocoPolysToMask
from util import datapoints
from util.misc import deepcopy


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        train=False,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.prepare = ConvertCocoPolysToMask()
        self._transforms = transforms
        self._transforms = self.update_dataset(self._transforms)
        self.train = train

        if train:
            self._coco_remove_images_without_annotations()
        
        # 过滤掉缺失或损坏的图像文件（训练和验证都适用）
        self._remove_missing_images()

    def update_dataset(self, transform):
        if isinstance(transform, (T.Compose, A.Compose)):
            processed_transforms = []
            for trans in transform.transforms:
                trans = self.update_dataset(trans)
                processed_transforms.append(trans)
            return type(transform)(processed_transforms)
        if hasattr(transform, "update_dataset"):
            transform.update_dataset(self)
        return transform

    def load_image(self, image_name):
        # after comparing the speed of PIL, torchvision and cv2,
        # cv2 is chosen as the default backend to load images,
        # uncomment the following code to switch among them.

        # image = Image.open(os.path.join(self.root, path)).convert('RGB')
        # image = torchvision.io.read_image(os.path.join(self.root, path))

        # To avoid deadlock between DataLoader and OpenCV
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # 规范化路径，处理Windows路径问题
        image_path = os.path.join(self.root, image_name)
        image_path = os.path.normpath(image_path)  # 规范化路径，处理双斜杠和混合斜杠
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # 尝试使用 cv2.imread 作为备用方案
        try:
            # 先尝试使用 np.fromfile 读取（支持中文路径）
            image_data = np.fromfile(image_path, dtype=np.uint8)
            if len(image_data) == 0:
                raise ValueError(f"Image file is empty: {image_path}")
            image = cv2.imdecode(image_data, -1)
            if image is None:
                # 如果 imdecode 失败，尝试使用 cv2.imread
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to decode image: {image_path}")
        except Exception as e:
            # 如果都失败，尝试使用 PIL 作为最后的备用方案
            try:
                from PIL import Image
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # PIL 是 RGB，需要转为 BGR
            except Exception as e2:
                raise RuntimeError(f"Failed to load image {image_path}: {e}, fallback also failed: {e2}")
        
        # 检查图像是否为空
        if image is None or image.size == 0:
            raise ValueError(f"Loaded image is empty: {image_path}")
        
        # 转换为 RGB 并转置
        if len(image.shape) == 2:
            # 灰度图，转换为3通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA 图像，转换为 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # BGR 图像，转换为 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = image.transpose(2, 0, 1)
        return image

    def get_image_id(self, item: int):
        if hasattr(self, "indices"):
            item = self.indices[item]
        image_id = self.ids[item]
        return image_id

    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def data_augmentation(self, image, target):
        # preprocess
        image = datapoints.Image(image)
        bounding_boxes = datapoints.BoundingBox(
            target["boxes"],
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=image.shape[-2:],
        )
        labels = target["labels"]
        if self._transforms is not None:
            image, bounding_boxes, labels = self._transforms(image, bounding_boxes, labels)

        return image.data, bounding_boxes.data, labels

    def __getitem__(self, item):
        # 添加重试机制，处理文件损坏的情况（文件缺失已在初始化时过滤）
        max_retries = 10  # 增加重试次数，因为可能连续多个文件损坏
        original_item = item
        for retry in range(max_retries):
            try:
                image, target = self.load_image_and_target(item)
                image, target["boxes"], target["labels"] = self.data_augmentation(image, target)
                return deepcopy(image), deepcopy(target)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                if retry < max_retries - 1:
                    # 尝试下一个样本（使用模运算确保不越界）
                    item = (item + 1) % len(self)
                    # 如果绕了一圈回到起点，说明所有样本都有问题
                    if item == original_item:
                        raise RuntimeError(f"All images in dataset appear to be invalid. Last error: {e}")
                    continue
                else:
                    # 如果所有重试都失败，抛出错误
                    raise RuntimeError(f"Failed to load image after {max_retries} retries. Last error: {e}")

    def __len__(self):
        return len(self.indices) if hasattr(self, "indices") else len(self.ids)

    def _coco_remove_images_without_annotations(self, cat_list=None):
        def _has_only_empty_bbox(anno):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

        def _count_visible_keypoints(anno):
            return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

        min_keypoints_per_image = 10

        def _has_valid_annotation(anno):
            # if it's empty, there is no annotation
            if len(anno) == 0:
                return False
            # if all boxes have close to zero area, there is no annotation
            if _has_only_empty_bbox(anno):
                return False
            # keypoints task have a slight different critera for considering
            # if an annotation is valid
            if "keypoints" not in anno[0]:
                return True
            # for keypoint detection tasks, only consider valid images those
            # containing at least min_keypoints_per_image
            if _count_visible_keypoints(anno) >= min_keypoints_per_image:
                return True
            return False

        ids = []
        for ds_idx, img_id in enumerate(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if cat_list:
                anno = [obj for obj in anno if obj["category_id"] in cat_list]
            if _has_valid_annotation(anno):
                ids.append(ds_idx)

        self.indices = ids

    def _remove_missing_images(self):
        """移除缺失或损坏的图像文件"""
        import logging
        logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        
        valid_ids = []
        missing_count = 0
        
        # 获取要检查的ID列表
        ids_to_check = self.indices if hasattr(self, "indices") else list(range(len(self.ids)))
        
        for ds_idx in ids_to_check:
            img_id = self.ids[ds_idx]
            try:
                # 获取图像文件名
                image_info = self.coco.loadImgs([img_id])[0]
                image_name = image_info["file_name"]
                
                # 检查文件是否存在
                image_path = os.path.join(self.root, image_name)
                image_path = os.path.normpath(image_path)
                
                if not os.path.exists(image_path):
                    missing_count += 1
                    if missing_count <= 5:  # 只打印前5个缺失文件的警告
                        logger.warning(f"Image file not found, skipping: {image_path}")
                    continue
                
                # 尝试快速检查文件是否可读（不实际加载图像）
                try:
                    if os.path.getsize(image_path) == 0:
                        missing_count += 1
                        if missing_count <= 5:
                            logger.warning(f"Image file is empty, skipping: {image_path}")
                        continue
                except OSError:
                    missing_count += 1
                    if missing_count <= 5:
                        logger.warning(f"Cannot access image file, skipping: {image_path}")
                    continue
                
                valid_ids.append(ds_idx)
            except Exception as e:
                missing_count += 1
                if missing_count <= 5:
                    logger.warning(f"Error checking image {img_id}, skipping: {e}")
                continue
        
        if missing_count > 0:
            logger.info(f"Filtered out {missing_count} missing or invalid images. Remaining: {len(valid_ids)} images")
        
        # 更新indices
        if hasattr(self, "indices"):
            self.indices = valid_ids
        else:
            # 如果没有indices，创建一个新的
            self.indices = valid_ids


class Object365Detection(CocoDetection):
    def load_image_and_target(self, item: int):
        image_id = self.get_image_id(item)
        # load images and annotations
        image_name = self.coco.loadImgs([image_id])[0]["file_name"]
        # NOTE: Only for object 365
        image_name = os.path.join(*image_name.split(os.sep)[-2:])
        if self.train:
            image_name = os.path.join("images/train", image_name)
        else:
            image_name = os.path.join("images/val", image_name)
        image = self.load_image(image_name)
        target = self.coco.loadAnns(self.coco.getAnnIds([image_id]))
        target = dict(image_id=image_id, annotations=target)
        image, target = self.prepare((image, target))
        return image, target

    def __getitem__(self, item):
        try:
            image, target = self.load_image_and_target(item)
        except:
            item += 1
            image, target = self.load_image_and_target(item)
        image, target["boxes"], target["labels"] = self.data_augmentation(image, target)

        return deepcopy(image), deepcopy(target)
