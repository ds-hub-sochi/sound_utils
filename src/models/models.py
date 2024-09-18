import torch
import torch.nn.functional as F
import torchaudio
import transformers
from torch import nn
from speechbrain.inference.classifiers import EncoderClassifier  # 


class SpectrogramBasedClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
    ):
        super().__init__()

        self._model: nn.Module = backbone

        if hasattr(
            self._model,
            'conv1',
        ):
            self._model.conv1 = nn.Conv2d(
                1,
                self._model.conv1.out_channels,
                kernel_size=self._model.conv1.kernel_size[0],
                stride=self._model.conv1.stride[0],
                padding=self._model.conv1.padding[0],
                dtype=torch.float32,
            )
        elif hasattr(
            self._model,
            'conv_proj',
        ):
            self._model.conv_proj = nn.Conv2d(
                1,
                self._model.conv_proj.out_channels,
                kernel_size=self._model.conv_proj.kernel_size[0],
                stride=self._model.conv_proj.stride[0],
                padding=self._model.conv_proj.padding[0],
                dtype=torch.float32,
            )
        elif hasattr(
            self._model,
            'features',
        ):
            self._model.features[0][0] = nn.Conv2d(
                1,
                self._model.features[0][0].out_channels,
                kernel_size=self._model.features[0][0].kernel_size[0],
                stride=self._model.features[0][0].stride[0],
                padding=self._model.features[0][0].padding[0],
                dtype=torch.float32,
            )

        if hasattr(
            self._model,
            'fc',
        ):
            num_features: int = self._model.fc.in_features
            self._model.fc = nn.Linear(
                num_features,
                num_classes,
                dtype=torch.float32,
            )
        elif hasattr(
            self._model,
            'heads',
        ):
            num_features = self._model.heads.head.in_features
            self._model.heads = nn.Linear(
                num_features,
                num_classes,
                dtype=torch.float32,
            )
        elif hasattr(
            self._model,
            'classifier',
        ):
            num_features = self._model.classifier[1].in_features
            self._model.classifier[1] = nn.Linear(
                num_features,
                num_classes,
                dtype=torch.float32,
            )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self._model(input_tensor)


class WavBasedClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        feature_extractor: nn.Module,
        n_first_encoder_layers_to_use: int,
    ):
        super().__init__()

        self.feature_extractor: torchaudio.models.Wav2Vec2Model = feature_extractor

        hidden_size: int = 0
        if hasattr(
            feature_extractor,
            'encoder',
        ):
            hidden_size = feature_extractor.encoder.transformer.layers[0].attention.k_proj.out_features
            self.feature_extractor.encoder.transformer.layers = self.feature_extractor.encoder.transformer.layers[:n_first_encoder_layers_to_use]
        elif hasattr(
            feature_extractor,
            'model',
        ):
            hidden_size = feature_extractor.model.encoder.transformer.layers[0].attention.k_proj.out_features
            self.feature_extractor.model.encoder.transformer.layers = self.feature_extractor.model.encoder.transformer.layers[:n_first_encoder_layers_to_use]

        self.linear: nn.Linear = nn.Linear(
            hidden_size,
            n_classes,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        features: torch.Tensor = self._get_embeddings(input_tensor)
        logits: torch.Tensor = self.linear(features)

        return logits

    def _get_embeddings(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        embeddings: torch.Tensor = self.feature_extractor(input_tensor)[0].max(axis=1).values

        return F.normalize(embeddings)


class ASTBasedClassifier(nn.Module):
    def __init__(
        self,
        model: transformers.ASTForAudioClassification,
        n_classes: int,
    ):
        super().__init__()

        in_features: int = model.classifier.dense.in_features
        
        self._model: transformers.ASTForAudioClassification = model
        self._model.classifier.dense = nn.Linear(
            in_features,
            n_classes,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self._model(input_tensor).logits


class WhisperBasedClassifier(nn.Module):
    def __init__(
        self,
        model: transformers.WhisperForAudioClassification,
        n_classes: int,
        n_first_encoders_to_use: int,
    ):
        super().__init__()

        in_features: int = model.classifier.in_features
        
        self._model: transformers.WhisperForAudioClassification = model
        self._model.encoder.layers = self._model.encoder.layers[:n_first_encoders_to_use]
        self._model.classifier = nn.Linear(
            in_features,
            n_classes,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self._model(input_tensor).logits


class SpeechBrainWrapper(EncoderClassifier):
    def encode_batch(
        self,
        wavs: torch.Tensor,
    ) -> torch.Tensor:
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        wav_lens: torch.Tensor = torch.ones(wavs.shape[0], device=self.device)

        wavs = wavs.float()

        feats: torch.Tensor = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        feats = feats.mean(axis=-1)  # type: ignore
        embeddings: torch.Tensor = self.mods.embedding_model(
            feats,
            wav_lens,
        )

        return embeddings

    def classify_batch(
        self,
        wavs: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.encode_batch(wavs)

        return self.mods.classifier(emb).squeeze(1)

        return logits

    def forward(
        self,
        wavs: torch.Tensor,
    ):
        return self.classify_batch(wavs)


class SpeechBrainBasedClassifier(SpeechBrainWrapper):
    def __init__(
        self,
        model: SpeechBrainWrapper,
        n_classes: int,
    ):
        super().__init__()
        
        self._model: SpeechBrainWrapper = model
        
        in_features = self._model.mods.classifier.out.w.in_features
        self._model.mods.classifier.out.w = torch.nn.Linear(
            in_features,
            n_classes,
        )
        self._model.mods.classifier.softmax = torch.nn.Identity()

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self._model(input_tensor)
