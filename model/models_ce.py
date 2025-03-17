import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


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
            # replacement_mask = self.rng.binomial(n = 1, p = replacement_probs[indices])
            logits = torch.log(replacement_probs[indices] + 1e-10)
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-10) + 1e-10
            )
            gumbel_softmax_sample = F.softmax((logits + gumbel_noise) / 0.1, dim=-1)
            replacement_mask = (gumbel_softmax_sample > 0.5).float()

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


@dataclass
class RationaleExtractorFactory:
    tokenizer: PreTrainedTokenizerBase
    device: str
    data_path: Optional[str] = None
    seed: Optional[int] = None

    def create_extractor(self, inject_noise):
        if inject_noise:
            return RandomNoisyRationaleExtractor(
                tokenizer=self.tokenizer,
                device=self.device,
                data_path=self.data_path,
                seed=self.seed,
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
