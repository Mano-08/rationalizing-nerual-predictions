import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

import math
from typing import Literal, Optional


@dataclass
class NoiseScheduler:
    """Scheduler for adaptive noise injection during training.

    Supports different scheduling strategies for gradually reducing noise level
    from high to low as training progresses.
    """

    noise_high: float  # Starting noise level
    noise_low: float  # Ending noise level
    total_steps: int  # Total number of training steps/epochs
    strategy: Literal["cosine", "exponential", "linear"] = "cosine"
    warmup_steps: int = 0  # Optional warmup period with max noise

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
        # During warmup, use maximum noise
        if current_step < self.warmup_steps:
            return self.noise_high

        # Adjust step to account for warmup
        adjusted_step = current_step - self.warmup_steps
        adjusted_total = self.total_steps - self.warmup_steps

        # Ensure we don't divide by zero if total_steps equals warmup_steps
        if adjusted_total <= 0:
            return self.noise_low

        # Normalize progress between 0 and 1
        progress = min(adjusted_step / adjusted_total, 1.0)

        if self.strategy == "cosine":
            # Cosine annealing: smooth transition with slower decay in the middle
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.noise_low + (self.noise_high - self.noise_low) * cosine_decay

        elif self.strategy == "exponential":
            # Exponential decay: faster initial decay, then gradual approach to minimum
            decay_rate = 3  # Controls decay speed, higher = faster decay
            exponential_decay = math.exp(-decay_rate * progress)
            return (
                self.noise_low + (self.noise_high - self.noise_low) * exponential_decay
            )

        else:  # linear
            # Linear decay: constant rate of decrease
            return self.noise_high - progress * (self.noise_high - self.noise_low)


@dataclass
class RationaleExtractor:
    tokenizer: PreTrainedTokenizerBase
    device: str

    def tokenize(self, query):
        return self.tokenizer(text=query, is_split_into_words=True, return_tensors="pt")

    def preprocess_mask(self, batch, hard_mask):
        hard_mask = hard_mask.squeeze()
        # Mask PAD tokens
        hard_mask.masked_fill_(
            batch.tokenized_examples.input_ids[:, 1:] == self.tokenizer.pad_token_id,
            False,
        )  # [:, 1:] - ignore the CLS token
        # Unmask SEP tokens
        hard_mask.masked_fill_(
            batch.tokenized_examples.input_ids[:, 1:] == self.tokenizer.sep_token_id,
            True,
        )
        return hard_mask

    def extract(self, batch, hard_mask):
        input_ids = []
        attention_mask = []
        # Extract rationales and combine with tokenized queries
        for query, ids, mask in zip(
            batch.examples.queries, batch.tokenized_examples.input_ids, hard_mask
        ):
            # Add tokenized query as base: CLS query tokens SEP
            tokenized_query = self.tokenize(query)
            input_ids.append(tokenized_query.input_ids.to(self.device))
            attention_mask.append(tokenized_query.attention_mask.to(self.device))
            # Append rationale: CLS query tokens SEP rationale tokens SEP
            rationale_ids = (
                ids[1:].masked_select(mask).unsqueeze(0)
            )  # [1:] - ignore the CLS token
            input_ids[-1] = torch.cat(
                (input_ids[-1], ids[1:].masked_select(mask).unsqueeze(0)), 1
            )
            attention_mask[-1] = torch.cat(
                (
                    attention_mask[-1],
                    torch.ones_like(rationale_ids, device=self.device),
                ),
                1,
            )
        return input_ids, attention_mask

    def pad(self, input_ids, attention_mask):
        # Get max seq length to pad to
        max_len = max([ids.shape[1] for ids in input_ids])
        # Pad
        for i in range(len(input_ids)):
            # Pad if necessary: CLS query tokens SEP rationale tokens SEP PAD [to max_len]
            if input_ids[i].shape[1] < max_len:
                pad = torch.full(
                    size=(1, max_len - input_ids[i].shape[1]),
                    fill_value=self.tokenizer.pad_token_id,
                    device=self.device,
                )
                input_ids[i] = torch.cat((input_ids[i], pad), 1)
                attention_mask[i] = torch.cat(
                    (attention_mask[i], torch.zeros_like(pad)), 1
                )
        return input_ids, attention_mask

    def to_hf_dict(self, input_ids, attention_mask):
        input_ids = torch.cat(input_ids, 0).to(self.device)
        attention_mask = torch.cat(attention_mask, 0).to(self.device)
        token_type_ids = torch.zeros_like(input_ids).to(self.device)
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

    def extract_from_mask(self, batch, hard_mask):
        hard_mask = self.preprocess_mask(batch, hard_mask)
        input_ids, attention_mask = self.extract(batch, hard_mask)
        input_ids, attention_mask = self.pad(input_ids, attention_mask)
        return self.to_hf_dict(input_ids, attention_mask)

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
    def extract_rationale(self, evidence, indices, replacement_mask):
        pass

    def get_examples(self, queries, rationales, sep_token):
        return [
            query + [sep_token] + rationale
            for query, rationale in zip(queries, rationales)
        ]

    def batch_tokenize(self, queries, rationales):
        text = self.get_examples(queries, rationales, self.tokenizer.sep_token)
        return self.tokenizer(
            text=text, is_split_into_words=True, padding=True, return_tensors="pt"
        ).to(self.device)

    def extract_from_mask_with_replacement(self, batch, hard_mask):
        # Mask PAD tokens
        hard_mask = (
            hard_mask.squeeze()
        )  # (batch_size, seq_len - CLS, 1) -> (batch_size, seq_len - CLS)
        hard_mask.masked_fill_(
            batch.tokenized_examples.input_ids[:, 1:] == self.tokenizer.pad_token_id,
            False,
        )  # (batch_size, seq_len - CLS, 1)
        # Mask SEP tokens - because we are retokenizing rationale batches
        hard_mask.masked_fill_(
            batch.tokenized_examples.input_ids[:, 1:] == self.tokenizer.sep_token_id,
            False,
        )  # (batch_size, seq_len - CLS, 1)
        # Extract rationales and pad
        rationales = []
        replacement_ratio_sum = 0
        for i, (evidence, replacement_probs, mask) in enumerate(
            zip(batch.examples.evidences, batch.replacement_probs, hard_mask)
        ):
            # Masked select
            indices = [
                x
                for x, is_masked in zip(batch.tokenized_examples.word_ids(i), mask)
                if x is not None and is_masked
            ]
            # Unique Consecutive
            indices = [
                x for i, x in enumerate(indices) if i == 0 or x != indices[i - 1]
            ]
            # Map to evidence indices
            indices = [x - batch.evidences_start_no_cls(i) for x in indices]
            # If no valid indices (selected tokens do not map to words such as
            # selecting only CLS/SEP/PAD tokens), select the first word
            if len(indices) == 0:
                indices = [0]
            # Sample replacement mask
            replacement_mask = self.rng.binomial(n=1, p=replacement_probs[indices])
            # Extract rationale
            rationale, replacement_ratio = self.extract_rationale(
                evidence=evidence, indices=indices, replacement_mask=replacement_mask
            )
            replacement_ratio_sum += replacement_ratio
            rationales.append(rationale)

        average_replacement_ratio = replacement_ratio_sum / len(
            batch.examples.evidences
        )

        tokenized_rationales = self.batch_tokenize(batch.examples.queries, rationales)
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

    def extract_rationale(self, evidence, indices, replacement_mask):
        rationale = np.array(evidence, dtype=self.vocab.dtype)[indices]
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
                noise_high=0.5,
                noise_low=0.2,
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

    def create_extractor(self, inject_noise, num_epochs):
        if inject_noise:
            noise_scheduler = NoiseScheduler(
                noise_high=0.5,  # Start with 50% noise
                noise_low=0.2,  # End with 5% noise
                total_steps=num_epochs,  # Total training steps/epochs
                strategy="cosine",  # You can also use "exponential" or "linear"
                warmup_steps=2,  # Optional: maintain high noise for first 2 epochs
            )
            return AdaptiveNoiseRationaleExtractor(
                tokenizer=self.tokenizer,
                device=self.device,
                data_path=self.data_path,
                seed=self.seed,
                noise_scheduler=noise_scheduler,
            )
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

    def get_atts_from_batch(self, token_att, batch, i):
        return token_att[i, batch.evidences_start_no_cls(i) :, :].squeeze().detach()

    def map_to_token_indices(self, evidence_indices, batch, i):
        return evidence_indices + batch.evidences_start_no_cls(i)

    # gets k = how many tokens to select
    def get_k(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_lens = torch.sum(input_ids != self.pad_token_id, dim=1)
        seq_lens[seq_lens > self.max_length - 1] = self.max_length - 1  # - CLS token
        k = torch.round(seq_lens * self.sparsity).type(torch.long)
        k[k < 1] = 1  # select at least 1 token
        return k

    def __call__(self, batch, token_att) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k(batch.tokenized_examples.input_ids)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = self.get_atts_from_batch(token_att, batch, i)
            _, evidence_indices = atts.topk(k=k[i], dim=-1)
            token_indices = self.map_to_token_indices(evidence_indices, batch, i)
            hard_mask[i, token_indices, :] = True
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
            [sent_mask[i] if i != -1 else -1 for i in word_ids], dtype=torch.long
        )  # word_ids[1:] -> - CLS token

    def get_word_ids_from_batch(self, batch, i):
        word_ids = batch.tokenized_examples.word_ids(i)[batch.evidences_start_cls(i) :]
        return [
            id - batch.evidences_start_cls(i) if id is not None else -1
            for id in word_ids
        ]

    def map_to_tokens_from_batch(self, batch, i):
        return self.map_to_tokens(
            batch.sentence_mask[i], self.get_word_ids_from_batch(batch, i)
        )

    def get_mean_sentence_val(self, tensor, sent_mask, i):
        mask = sent_mask == i
        return torch.sum(tensor[mask]) / torch.sum(mask)

    def aggregate_to_sent_level(self, tensor, sent_mask, num_sents, i):
        return [
            self.get_mean_sentence_val(tensor, sent_mask, i)
            for i in range(num_sents[i])
        ]

    def aggregate_atts_to_sent_level(self, atts, sent_mask, num_sents, i):
        return torch.tensor(self.aggregate_to_sent_level(atts, sent_mask, num_sents, i))

    def aggregate_counts_to_sent_level(self, counts, sent_mask, num_sents, i):
        return self.aggregate_to_sent_level(counts, sent_mask, num_sents, i)

    def map_to_evidence_indices(self, sent_indices, sent_mask, atts):
        return torch.tensor(
            [i for i in range(len(atts)) if torch.any(sent_indices == sent_mask[i])]
        )

    def __call__(self, batch, token_att) -> torch.Tensor:
        hard_mask = self.get_mask(token_att)
        k = self.get_k_sentences(batch.sentence_mask)
        num_sents = self.get_num_sentences(batch.sentence_mask)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = self.get_atts_from_batch(token_att, batch, i)
            sent_mask = self.map_to_tokens_from_batch(batch, i)
            sent_atts = self.aggregate_atts_to_sent_level(atts, sent_mask, num_sents, i)
            _, sent_indices = sent_atts.topk(k=k[i], dim=-1)
            evidence_indices = self.map_to_evidence_indices(
                sent_indices, sent_mask, atts
            )
            token_indices = self.map_to_token_indices(evidence_indices, batch, i)
            hard_mask[i, token_indices, :] = True
        return hard_mask

    def get_selected_sentences(self, batch, token_att):
        selected_sentences = []
        k = self.get_k_sentences(batch.sentence_mask)
        num_sents = self.get_num_sentences(batch.sentence_mask)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = self.get_atts_from_batch(token_att, batch, i)
            sent_mask = self.map_to_tokens_from_batch(batch, i)
            sent_atts = self.aggregate_atts_to_sent_level(atts, sent_mask, num_sents, i)
            _, sent_indices = sent_atts.topk(k=k[i], dim=-1)
            selected_sentences.append(sent_indices.tolist())
        return selected_sentences

    def get_selected_token_counts_by_sentences(self, batch, token_att):
        tcounts = []
        rcounts = []
        k = self.get_k_sentences(batch.sentence_mask)
        num_sents = self.get_num_sentences(batch.sentence_mask)
        # batch-first
        for i in range(token_att.shape[0]):
            atts = self.get_atts_from_batch(token_att, batch, i)
            sent_mask = self.map_to_tokens_from_batch(batch, i)
            _, evidence_indices = atts.topk(k=k[i], dim=-1)
            evidence_mask = torch.zeros_like(atts)
            evidence_mask[evidence_indices] = 1
            tcounts.append(
                self.aggregate_counts_to_sent_level(
                    evidence_mask, sent_mask, num_sents, i
                )
            )
            rcounts.append(
                self.aggregate_counts_to_sent_level(
                    ~evidence_mask, sent_mask, num_sents, i
                )
            )
        return tcounts, rcounts


@dataclass
class RationaleSelectorFactory:
    sparsity: float
    max_length: int
    pad_token_id: int
    device: str

    def create_selector(self, selection_method):
        if selection_method == "sentences":
            return TopkSentenceSelector(
                self.sparsity, self.max_length, self.pad_token_id, self.device
            )
        if selection_method == "words":
            return TopkSelector(
                self.sparsity, self.max_length, self.pad_token_id, self.device
            )
        raise ValueError("selection_method not one of ['sentences', 'words']")
