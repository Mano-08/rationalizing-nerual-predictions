import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerBase
import math
from typing import Literal, Optional


def js_div(P, Q):
    """Calculate Jensen-Shannon divergence between two probability distributions.
    Handles potential dimension mismatches by ensuring both tensors have same shape.
    """
    # Check dimensions and fix if necessary
    if P.size(1) != Q.size(1):
        # Determine which tensor needs padding
        if P.size(1) < Q.size(1):
            # Pad P to match Q's dimensions
            padding = torch.zeros(P.size(0), Q.size(1) - P.size(1), device=P.device)
            P = torch.cat([P, padding], dim=1)
        else:
            # Pad Q to match P's dimensions
            padding = torch.zeros(Q.size(0), P.size(1) - Q.size(1), device=Q.device)
            Q = torch.cat([Q, padding], dim=1)

    # Ensure distributions sum to 1 after padding
    P = F.normalize(P, p=1, dim=1)
    Q = F.normalize(Q, p=1, dim=1)

    # Calculate JS divergence
    M = (P + Q) / 2
    kl_div = lambda X, M: F.kl_div(X.log(), M, reduction="sum")
    return (kl_div(P, M) + kl_div(Q, M)) / 2


# def js_div(P, Q):
#     M = (P + Q) / 2
#     kl_div = lambda X, M: F.kl_div(X.log(), M, reduction="sum")
#     return (kl_div(P, M) + kl_div(Q, M)) / 2


@dataclass
class NoiseScheduler:
    """Scheduler for adaptive noise injection during training.

    Supports different scheduling strategies for gradually reducing noise level
    from high to low as training progresses.
    """

    noise_high: float
    noise_low: float
    total_steps: int
    strategy: Literal["cosine", "exponential", "linear"] = "cosine"
    warmup_steps: int = 0

    def __post_init__(self):
        assert (
            0 <= self.noise_low <= self.noise_high <= 1.0
        ), "Noise levels must be between 0 and 1"
        assert self.total_steps > 0, "Total steps must be positive"
        assert self.warmup_steps >= 0, "Warmup steps must be non-negative"

    def get_noise_level(self, current_step: int) -> float:
        """Calculate noise level for the current training step.

        Args:
            current_step: Current training step/epoch

        Returns:
            Current noise level between noise_high and noise_low
        """
        if current_step < self.warmup_steps:
            return self.noise_high

        adjusted_step = current_step - self.warmup_steps
        adjusted_total = self.total_steps - self.warmup_steps

        if adjusted_total <= 0:
            return self.noise_low

        progress = min(adjusted_step / adjusted_total, 1.0)

        if self.strategy == "cosine":
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.noise_low + (self.noise_high - self.noise_low) * cosine_decay

        elif self.strategy == "exponential":
            decay_rate = 3
            exponential_decay = math.exp(-decay_rate * progress)
            return (
                self.noise_low + (self.noise_high - self.noise_low) * exponential_decay
            )

        else:
            return self.noise_high - progress * (self.noise_high - self.noise_low)


class BlackBoxPredictor(nn.Module):
    def __init__(self, num_labels: int, model: str, freeze_encoder: bool):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.token_predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, 1),
        )
        self.predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels),
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        # get contextualized embeddings from a transformer-based encoder
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        # get hidden states without the CLS token
        hidden_states_no_cls = outputs[0][:, 1:, :]
        # use token classification head to generate token logits
        token_logits = self.token_predictor(hidden_states_no_cls).squeeze()
        # use softmax to get attention over tokens
        token_att = F.softmax(token_logits, -1).unsqueeze(-1)
        # generate context vector
        ctx_vec = torch.bmm(hidden_states_no_cls.transpose(1, 2), token_att).squeeze()
        # return predicted labels, per token probabilities P(z|x)
        return F.softmax(self.predictor(ctx_vec), -1), token_att

    def get_loss(self, att_pred, hard_pred, labels, proximity):
        return F.cross_entropy(att_pred, labels) + proximity * js_div(
            att_pred, hard_pred
        )


class RationalePredictor(nn.Module):
    def __init__(self, num_labels: int, model: str, freeze_encoder: bool):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels),
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return F.softmax(self.predictor(outputs[1]), -1)

    def get_loss(self, att_pred, hard_pred, labels, proximity):
        return F.cross_entropy(hard_pred, labels) + proximity * js_div(
            att_pred, hard_pred
        )


@dataclass
class RationaleExtractor:
    tokenizer: PreTrainedTokenizerBase
    device: str

    def update_step(self, step: int):
        pass

    def extract_from_mask(self, batch, hard_mask):
        # Add CLS tokens
        hard_mask = torch.cat(
            (torch.ones_like(hard_mask[:, 1]).unsqueeze(-1), hard_mask), 1
        ).squeeze()  # (batch_size, seq_len - CLS, 1) -> (batch_size, seq_len)
        # Mask PAD tokens
        hard_mask.masked_fill_(
            batch.reviews_tokenized.input_ids == self.tokenizer.pad_token_id, False
        )
        # Unmask SEP tokens - because we are taking rationale batches as is
        hard_mask.masked_fill_(
            batch.reviews_tokenized.input_ids == self.tokenizer.sep_token_id, True
        )
        # Get max seq length to pad to
        max_len = hard_mask.sum(dim=1).max()
        # Extract rationales and pad
        rationale_ids = []
        attention_mask = []
        for ids, mask in zip(batch.reviews_tokenized.input_ids, hard_mask):
            rationale_ids.append(ids.masked_select(mask).unsqueeze(0))
            attention_mask.append(
                torch.ones_like(rationale_ids[-1], device=self.device)
            )
            # pad if necessary
            if rationale_ids[-1].shape[1] < max_len:
                pad = torch.full(
                    size=(1, max_len - rationale_ids[-1].shape[1]),
                    fill_value=self.tokenizer.pad_token_id,
                    device=self.device,
                )
                rationale_ids[-1] = torch.cat((rationale_ids[-1], pad), 1)
                attention_mask[-1] = torch.cat(
                    (attention_mask[-1], torch.zeros_like(pad)), 1
                )

        # to tensors
        rationale_ids = torch.cat(rationale_ids, 0).to(self.device)
        attention_mask = torch.cat(attention_mask, 0).to(self.device)
        token_type_ids = torch.zeros_like(rationale_ids).to(self.device)
        return {
            "input_ids": rationale_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

    def __call__(self, batch, hard_mask):
        tokenized_rationales = self.extract_from_mask(batch, hard_mask)
        tokenized_remainders = self.extract_from_mask(batch, ~hard_mask)
        replacement_ratio = 0
        return tokenized_rationales, tokenized_remainders, replacement_ratio


@dataclass
class BaseNoisyRationaleExtractor(RationaleExtractor):
    data_path: str
    seed: Optional[int] = None

    def __post_init__(self):
        self.load_scored_vocab()
        self.rng = np.random.default_rng(self.seed)

    def load_scored_vocab(self):
        with open(
            os.path.join(self.data_path, "word_statistics", "scored_vocab.pkl"), "rb"
        ) as f:
            self.vocab, self.scores = pickle.load(f)

    @abstractmethod
    def extract_rationale(self, review, indices, replacement_mask):
        pass

    def extract_remainder(self, review, remainder_indices):
        return [review[i] for i in remainder_indices]

    def batch_tokenize(self, text):
        return self.tokenizer(
            text=text, is_split_into_words=True, padding=True, return_tensors="pt"
        ).to(self.device)

    def extract_from_mask_with_replacement(self, batch, hard_mask):
        # Mask PAD tokens
        hard_mask = (
            hard_mask.squeeze()
        )  # (batch_size, seq_len - CLS, 1) -> (batch_size, seq_len - CLS)
        hard_mask.masked_fill_(
            batch.reviews_tokenized.input_ids[:, 1:] == self.tokenizer.pad_token_id,
            False,
        )  # (batch_size, seq_len - CLS, 1)
        # Mask SEP tokens - because we are retokenizing rationale batches
        hard_mask.masked_fill_(
            batch.reviews_tokenized.input_ids[:, 1:] == self.tokenizer.sep_token_id,
            False,
        )  # (batch_size, seq_len - CLS, 1)
        # Extract rationales and pad
        rationales = []
        replacement_ratio_sum = 0
        for i, (review, replacement_probs, mask) in enumerate(
            zip(batch.reviews, batch.replacement_probs, hard_mask)
        ):
            # Masked select
            indices = [
                x
                for x, is_masked in zip(batch.reviews_tokenized.word_ids(i), mask)
                if x is not None and is_masked
            ]
            # Unique Consecutive
            indices = [
                x for i, x in enumerate(indices) if i == 0 or x != indices[i - 1]
            ]
            # If no valid indices (selected tokens do not map to words such as
            # selecting only CLS/SEP/PAD tokens), select the first word
            if len(indices) == 0:
                indices = [0]
            # Sample replacement mask
            replacement_mask = self.rng.binomial(n=1, p=replacement_probs[indices])
            # Extract rationale
            rationale, replacement_ratio = self.extract_rationale(
                review=review, indices=indices, replacement_mask=replacement_mask
            )
            replacement_ratio_sum += replacement_ratio
            rationales.append(rationale)

        average_replacement_ratio = replacement_ratio_sum / len(batch.reviews)

        tokenized_rationales = self.batch_tokenize(rationales)
        return tokenized_rationales, average_replacement_ratio

    def __call__(self, batch, hard_mask):
        tokenized_rationales, replacement_ratio = (
            self.extract_from_mask_with_replacement(batch, hard_mask)
        )
        tokenized_remainder = self.extract_from_mask(batch, ~hard_mask)
        return tokenized_rationales, tokenized_remainder, replacement_ratio


@dataclass
class RandomNoisyRationaleExtractor(BaseNoisyRationaleExtractor):
    def __post_init__(self):
        super().__post_init__()
        self.vocab = np.array(self.vocab)

    def extract_rationale(self, review, indices, replacement_mask):
        rationale = np.array(review, dtype=self.vocab.dtype)[indices]
        rationale[replacement_mask == 1] = self.rng.choice(
            self.vocab, size=replacement_mask.sum(), p=self.scores
        )
        return rationale.tolist(), replacement_mask.sum() / replacement_mask.shape[0]


# Modified BaseNoisyRationaleExtractor to use the noise scheduler
@dataclass
class AdaptiveNoiseRationaleExtractor(BaseNoisyRationaleExtractor):
    """Rationale extractor with adaptive noise injection.

    Uses a noise scheduler to dynamically adjust noise levels during training.
    """

    current_step: int = 0  # Current training step/epoch

    def __post_init__(self):
        super().__post_init__()
        # Ensure we have a noise scheduler
        if self.noise_scheduler is None:
            # Default noise scheduler with reasonable values
            self.noise_scheduler = NoiseScheduler(
                noise_high=0.3,
                noise_low=0.05,
                total_steps=10,  # Adjust based on your expected training length
                strategy="cosine",
            )
        self.vocab = np.array(self.vocab)

    def update_step(self, step: int):
        """Update the current training step/epoch."""
        self.current_step = step

    def get_current_noise_level(self) -> float:
        """Get the current noise level based on training progress."""
        return self.noise_scheduler.get_noise_level(self.current_step)

    def extract_rationale(self, evidence, indices, replacement_mask):
        # Get current noise level
        noise_level = self.get_current_noise_level()

        # Apply noise level to replacement_mask
        # This modifies the probability of replacement based on the current noise level
        adjusted_replacement_mask = replacement_mask * noise_level

        # Extract rationale with adjusted noise
        rationale = np.array(evidence, dtype=self.vocab.dtype)[indices]
        rationale[adjusted_replacement_mask == 1] = self.rng.choice(
            self.vocab, size=adjusted_replacement_mask.sum(), p=self.scores
        )

        # Calculate actual replacement ratio
        replacement_ratio = (
            adjusted_replacement_mask.sum() / adjusted_replacement_mask.shape[0]
        )

        return rationale.tolist(), replacement_ratio

    def __call__(self, batch, hard_mask):
        # Same implementation as parent, but will use our updated extract_rationale
        tokenized_rationales, replacement_ratio = (
            self.extract_from_mask_with_replacement(batch, hard_mask)
        )
        tokenized_remainder = self.extract_from_mask(batch, ~hard_mask)

        return tokenized_rationales, tokenized_remainder, replacement_ratio


@dataclass
class RationaleExtractorFactory:
    tokenizer: PreTrainedTokenizerBase
    device: str
    data_path: Optional[str] = None
    seed: Optional[int] = None
    adaptiveNoiseRationaleExtractor: Optional[AdaptiveNoiseRationaleExtractor] = None

    def update_step(self, step: int):
        if self.adaptiveNoiseRationaleExtractor is not None:
            self.adaptiveNoiseRationaleExtractor.update_step(step)

    def create_extractor(self, inject_noise, num_epochs):
        if inject_noise:
            noise_scheduler = NoiseScheduler(
                noise_high=0.3,
                noise_low=0.05,
                total_steps=num_epochs,
                strategy="cosine",
                warmup_steps=1,
            )
            adaptiveNoiseRationaleExtractor = AdaptiveNoiseRationaleExtractor(
                tokenizer=self.tokenizer,
                device=self.device,
                data_path=self.data_path,
                seed=self.seed,
                noise_scheduler=noise_scheduler,
            )
            self.adaptiveNoiseRationaleExtractor = adaptiveNoiseRationaleExtractor
            return adaptiveNoiseRationaleExtractor
        return RationaleExtractor(tokenizer=self.tokenizer, device=self.device)


@dataclass
class TopkSelector:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str

    # gets empty mask
    def get_mask(self, token_att):
        return torch.zeros_like(
            token_att, dtype=torch.bool, requires_grad=False, device=self.device
        )

    # gets k = how many tokens to select
    def get_k(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_lens = torch.sum(input_ids != self.pad_token_id, dim=1)
        seq_lens[seq_lens > self.max_length - 1] = self.max_length - 1  # - CLS token
        k = torch.round(seq_lens * self.sparsity).type(torch.long)
        k[k < 1] = 1  # select at least 1 token
        return k

    def __call__(
        self, token_att: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k(input_ids)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = token_att[i, :, :].squeeze().detach()
            _, indices = atts.topk(k=k[i], dim=-1)
            hard_mask[i, indices, :] = True
        return hard_mask


@dataclass
class TopkContiguousSpanSelector(TopkSelector):
    def __call__(
        self, token_att: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k(input_ids)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = token_att[i, :, :].unsqueeze(0).swapdims(2, 1)
            filter = torch.ones((1, 1, k[i]), device=self.device)
            start = F.conv1d(atts, filter).argmax()
            hard_mask[i, start : start + k[i], :] = True
        return hard_mask


@dataclass
class TopkSentenceSelector(TopkSelector):
    def get_num_sentences(self, sentence_mask: list):
        # batch-first - select last sentence index (= num_sentences - 1) from each seq in a batch
        return [sent_mask[-1] + 1 for sent_mask in sentence_mask]

    def get_k_sentences(self, sentence_mask: list):
        num_sents = torch.tensor(self.get_num_sentences(sentence_mask))
        k_sentences = torch.round(num_sents * self.sparsity).type(torch.long)
        k_sentences[k_sentences < 1] = 1  # select at least 1 sentence
        return k_sentences

    def map_to_tokens(self, sent_mask: list, word_ids: list):
        return torch.tensor(
            [sent_mask[i] if i is not None else -1 for i in word_ids[1:]],
            dtype=torch.long,
        )  # word_ids[1:] -> - CLS token

    def __call__(
        self, token_att: torch.Tensor, reviews_tokenized, sentence_mask: list
    ) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k_sentences(sentence_mask)
        num_sents = self.get_num_sentences(sentence_mask)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = token_att[i, :, :].squeeze().detach()
            sent_mask = self.map_to_tokens(
                sentence_mask[i], reviews_tokenized.word_ids(i)
            )
            sent_atts = torch.tensor(
                [torch.sum(atts[sent_mask == i]) for i in range(num_sents[i])]
            )
            _, sent_indices = sent_atts.topk(k=k[i], dim=-1)
            indices = torch.tensor(
                [i for i in range(len(atts)) if torch.any(sent_indices == sent_mask[i])]
            )
            hard_mask[i, indices, :] = True
        return hard_mask


@dataclass
class SelectorFactory:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str

    def create_selector(self, selection_method):
        if selection_method == "words":
            return TopkSelector(
                sparsity=self.sparsity,
                max_length=self.max_length,
                pad_token_id=self.pad_token_id,
                device=self.device,
            )
        elif selection_method == "span":
            return TopkContiguousSpanSelector(
                sparsity=self.sparsity,
                max_length=self.max_length,
                pad_token_id=self.pad_token_id,
                device=self.device,
            )
        raise ValueError(f"Unknown selection method {selection_method}")
