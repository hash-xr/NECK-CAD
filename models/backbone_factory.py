import abc
import logging
import torch
import torch.nn as nn
from typing import Dict, Type

logger = logging.getLogger("NECK-CAD.BackboneFactory")


class BaseBackbone(nn.Module, abc.ABC):
    """Abstract Base Class for all NECK-CAD feature extraction backbones."""

    def __init__(self, model_name: str, embedding_dim: int):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts normalised feature vectors from input image tensors.

        Args:
            x (torch.Tensor): Image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature embedding matrix of shape (B, embedding_dim).
        """
        pass

    def get_embedding_dim(self) -> int:
        """Returns feature vector dimension."""
        return self.embedding_dim


class MockBackbone(BaseBackbone):
    """Synthetic backbone for offline debugging and test suites."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__(model_name="mock", embedding_dim=embedding_dim)
        # Fixed linear projection layer to satisfy PyTorch gradient tracking
        self.dummy_layer = nn.Linear(3 * 224 * 224, embedding_dim)
        logger.info("Initialised MockBackbone for rapid testing.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flattened = x.view(batch_size, -1)
        # Resize or pad flattened tensor to fit dummy linear layer
        if flattened.shape[1] != 3 * 224 * 224:
            flattened = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            ).view(batch_size, -1)
        features = self.dummy_layer(flattened)
        return torch.nn.functional.normalize(features, p=2, dim=1)


class ResNet50Backbone(BaseBackbone):
    """ResNet-50 feature extractor pre-trained on ImageNet."""

    def __init__(self, pretrained: bool = True):
        super().__init__(model_name="resnet50", embedding_dim=2048)
        import torchvision.models as models

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_model = models.resnet50(weights=weights)
        # Strip final classification head to yield raw 2048-D feature map
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        logger.info(f"Initialised ResNet50Backbone (pretrained={pretrained}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)  # Shape: (B, 2048, 1, 1)
        features = torch.flatten(features, 1)  # Shape: (B, 2048)
        return torch.nn.functional.normalize(features, p=2, dim=1)


class PhikonBackbone(BaseBackbone):
    """Owkin Phikon-v2 ViT-L/16 feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__(model_name="phikon", embedding_dim=1024)
        if pretrained:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained("owkin/phikon-v2")
        else:
            from transformers import AutoConfig, AutoModel
            config = AutoConfig.from_pretrained("owkin/phikon-v2")
            self.encoder = AutoModel.from_config(config)
        logger.info("Initialised PhikonBackbone (owkin/phikon-v2).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=x)
        # Extract CLS token representation from final hidden state
        cls_features = outputs.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls_features, p=2, dim=1)


class UNIBackbone(BaseBackbone):
    """MahmoodLab UNI pathology foundation model."""

    def __init__(self, pretrained: bool = True):
        super().__init__(model_name="uni", embedding_dim=1024)
        import timm

        model_key = "hf-hub:MahmoodLab/UNI"
        self.encoder = timm.create_model(
            model_key, pretrained=pretrained, init_values=1e-5, dynamic_img_size=True
        )
        logger.info("Initialised UNIBackbone (MahmoodLab/UNI).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return torch.nn.functional.normalize(features, p=2, dim=1)


class VirchowBackbone(BaseBackbone):
    """Paige.ai Virchow2 ViT-H/14 pathology foundation model."""

    def __init__(self, pretrained: bool = True):
        super().__init__(model_name="virchow", embedding_dim=1280)
        import timm

        model_key = "hf-hub:paige-ai/Virchow2"
        self.encoder = timm.create_model(
            model_key, pretrained=pretrained, mlp_layer=timm.layers.mlp.SwiGLU, act_layer=torch.nn.SiLU
        )
        logger.info("Initialised VirchowBackbone (paige-ai/Virchow2).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder(x)  # Shape: (B, 1280)
        return torch.nn.functional.normalize(output, p=2, dim=1)


class BackboneFactory:
    """Central factory for instantiating model feature extractors."""

    _registry: Dict[str, Type[BaseBackbone]] = {
        "mock": MockBackbone,
        "resnet50": ResNet50Backbone,
        "phikon": PhikonBackbone,
        "uni": UNIBackbone,
        "virchow": VirchowBackbone,
    }

    @classmethod
    def register_backbone(cls, name: str, backbone_cls: Type[BaseBackbone]):
        """Registers a custom backbone implementation into the factory."""
        cls._registry[name.lower()] = backbone_cls
        logger.info(f"Registered custom backbone: {name}")

    @classmethod
    def create(cls, model_name: str, pretrained: bool = True, **kwargs) -> BaseBackbone:
        """
        Instantiates and returns the requested vision backbone.

        Args:
            model_name (str): Backbone key ('mock', 'resnet50', 'phikon', 'uni', 'virchow').
            pretrained (bool): Whether to load pre-trained weights.

        Returns:
            BaseBackbone: Ready-to-use PyTorch backbone module.
        """
        key = model_name.lower()
        if key not in cls._registry:
            valid_keys = list(cls._registry.keys())
            raise ValueError(f"Unknown backbone '{model_name}'. Valid choices are: {valid_keys}")

        backbone_cls = cls._registry[key]
        if key == "mock":
            return backbone_cls(**kwargs)
        return backbone_cls(pretrained=pretrained, **kwargs) #type: ignore