import torch
from torchvision import transforms


class HypertokImageProcessor:
    def __init__(self, size: int):
        self.size = size
        self.crop_size = {"height": size, "width": size}
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]
        self._tfm = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

    def preprocess(self, image, return_tensors="pt"):
        if isinstance(image, list):
            tensor_list = [self._tfm(img) for img in image]
            return {"pixel_values": torch.stack(tensor_list, dim=0)}
        tensor = self._tfm(image)
        return {"pixel_values": tensor.unsqueeze(0)}
