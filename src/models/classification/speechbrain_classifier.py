import torch
from speechbrain.inference.classifiers import EncoderClassifier  # pylint: disable=[import-error]
from torch import nn


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

    def forward(
        self,
        wavs: torch.Tensor,
    ):
        return self.classify_batch(wavs)


class SpeechBrainBasedClassifier(nn.Module):
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
