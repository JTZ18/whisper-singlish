import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
import evaluate


class WhisperFinetuning(pl.LightningModule):
    def __init__(self, lr, whisper_model="tiny"):
        super().__init__()
        self.lr = lr
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{whisper_model}")
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_model}")
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.wer_metric = evaluate.load("wer")

    def training_step(self, batch, batch_idx):
        encoder_input = batch[0]["input_features"]
        decoder_labels = batch[1]["labels"]

        out = self.model(
            input_features=encoder_input,
            labels=decoder_labels,
        )
        loss = out["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        encoder_input = batch[0]["input_features"]
        decoder_labels = batch[1]["labels"]

        out = self.model(
            input_features=encoder_input,
            labels=decoder_labels,
        )
        loss = out["loss"]
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Generate predictions for WER calculation
        predicted_ids = self.model.generate(encoder_input)
        transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        references = self.processor.batch_decode(decoder_labels, skip_special_tokens=True)

        wer = self.wer_metric.compute(predictions=transcriptions, references=references)
        self.log("val_wer", wer, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss, "val_wer": wer}

    def test_step(self, batch, batch_idx):
        encoder_input = batch[0]["input_features"]
        decoder_labels = batch[1]["labels"]

        out = self.model(
            input_features=encoder_input,
            labels=decoder_labels,
        )
        loss = out["loss"]
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Generate predictions for WER calculation
        predicted_ids = self.model.generate(encoder_input)
        transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        references = self.processor.batch_decode(decoder_labels, skip_special_tokens=True)

        wer = self.wer_metric.compute(predictions=transcriptions, references=references)
        self.log("test_wer", wer, on_epoch=True, prog_bar=True, logger=True)

        return {"test_loss": loss, "test_wer": wer}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    pass
