import json
import os
from argparse import ArgumentParser

import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from termcolor import colored
from tqdm.auto import tqdm
from transformers import AutoTokenizer, logging

from models import (
    BlackBoxPredictor,
    RationaleExtractor,
    RationaleExtractorFactory,
    RationalePredictor,
)
from models_ce import (
    RationaleExtractor,
    RationaleExtractorFactory,
    RationaleSelectorFactory,
    TopkSentenceSelector,
)
from datasets_ce import DataLoaderFactory

logging.set_verbosity_error()

MULTIRC_PATH = os.path.join("..", "..", "rnp_multirc", "original")
FEVER_PATH = os.path.join("..", "..", "rnp_fever", "original")
DATAPATH = {"multirc": MULTIRC_PATH, "fever": FEVER_PATH}


def parse_args():
    parser = ArgumentParser()
    # Whether to train and/or evaluate
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    # Whether to inject noise
    parser.add_argument("--inject_noise", action="store_true")
    # Magnitude of augmentation hyperparameter
    parser.add_argument("--noise_p", type=float, default=0.1)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Optimizer BB: pytorch optim.Adam defaults
    parser.add_argument("--bb_lr", type=float, default=2e-5)
    # Optimizer RP pytorch optim.Adam defaults
    parser.add_argument("--rp_lr", type=float, default=2e-5)
    # Freeze BERT weights
    parser.add_argument("--freeze_encoder_bb", action="store_true")
    parser.add_argument("--freeze_encoder_rp", action="store_true")
    # Training
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=-1)
    # Model proximity hyperparameter
    parser.add_argument("--proximity", type=float, default=0.1)
    # Model
    parser.add_argument(
        "--save_path", type=str, default=os.path.join("trained", "ours")
    )
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    # Selection method
    parser.add_argument(
        "--selection_method", type=str, default="words", choices=["words", "sentences"]
    )
    # Rationale Extraction hyperparameter
    parser.add_argument("--sparsity", type=float, default=0.2)
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="multirc", choices=["multirc", "fever"]
    )
    parser.add_argument("--data_path", type=str, default=None)
    # Eval-related
    # Compare model-generated and hand-labeled rationales
    parser.add_argument("--show_detail", action="store_true")
    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = DATAPATH[args.dataset]
    return args


def main(args):
    if not args.train and not args.evaluate:
        print("Must append flag --train or --evaluate")
        return

    checkpoint_dir = os.path.join(args.save_path, "checkpoints")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    bb_model = BlackBoxPredictor(
        num_labels=3, model=args.model, freeze_encoder=args.freeze_encoder_bb
    ).to(args.device)
    print(f"Black Box Predictor: {get_num_params(bb_model)} parameters")

    rp_model = RationalePredictor(
        num_labels=2, model=args.model, freeze_encoder=args.freeze_encoder_rp
    ).to(args.device)
    print(f"Rationale Predictor: {get_num_params(rp_model)} parameters")

    rationale_selector = RationaleSelectorFactory(
        args.sparsity, args.max_length, tokenizer.pad_token_id, args.device
    ).create_selector(args.selection_method)

    rationale_extractor = RationaleExtractorFactory(
        tokenizer, args.device, args.data_path
    ).create_extractor(args.inject_noise, args.num_epochs)

    if args.train:
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_loader = DataLoaderFactory(
            data_path=args.data_path,
            noise_p=args.noise_p,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            max_length=args.max_length,
            shuffle=True,
        ).create_dataloader("train", args.inject_noise)
        valid_loader = DataLoaderFactory(
            data_path=args.data_path,
            noise_p=args.noise_p,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            max_length=args.max_length,
            shuffle=True,
        ).create_dataloader("valid", args.inject_noise)

        bb_optimizer = optim.Adam(bb_model.parameters(), args.bb_lr)
        rp_optimizer = optim.Adam(rp_model.parameters(), args.rp_lr)

        validation_rationale_extractor = RationaleExtractor(tokenizer, args.device)

        train(
            bb_model=bb_model,
            bb_optimizer=bb_optimizer,
            rp_model=rp_model,
            rp_optimizer=rp_optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            eval_every=len(train_loader) // 2,
            device=args.device,
            proximity=args.proximity,
            num_epochs=args.num_epochs,
            patience=args.patience,
            checkpoint_dir=checkpoint_dir,
            rationale_selector=rationale_selector,
            rationale_extractor=rationale_extractor,
            validation_rationale_extractor=validation_rationale_extractor,
        )

    if args.evaluate:
        test_loader = DataLoaderFactory(
            data_path=args.data_path,
            noise_p=args.noise_p,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            max_length=args.max_length,
            shuffle=False,
        ).create_dataloader("test", args.inject_noise)

        sentence_selector = TopkSentenceSelector(
            args.sparsity, args.max_length, tokenizer.pad_token_id, args.device
        )
        test_rationale_extractor = RationaleExtractor(tokenizer, args.device)

        evaluate(
            bb_model=bb_model,
            rp_model=rp_model,
            test_loader=test_loader,
            device=args.device,
            show_detail=args.show_detail,
            rationale_selector=rationale_selector,
            rationale_extractor=test_rationale_extractor,
            sentence_selector=sentence_selector,
            checkpoint_dir=checkpoint_dir,
            result_path=args.save_path,
        )


def train(
    bb_model,
    bb_optimizer,
    rp_model,
    rp_optimizer,
    train_loader,
    valid_loader,
    eval_every,
    device,
    proximity,
    num_epochs,
    patience,
    checkpoint_dir,
    rationale_selector,
    rationale_extractor,
    validation_rationale_extractor,
):

    with tqdm(total=num_epochs * len(train_loader)) as pb:

        # Initialize statistics
        bb_running_train_loss = 0.0
        bb_best_train_loss = float("Inf")
        rp_running_train_loss = 0.0
        rp_best_train_loss = float("Inf")
        bb_running_valid_loss = 0.0
        bb_best_valid_loss = float("Inf")
        rp_running_valid_loss = 0.0
        rp_best_valid_loss = float("Inf")
        running_train_replace_ratio = 0.0
        running_valid_replace_ratio = 0.0
        global_step = 0
        metrics = []
        patience_left = patience

        # training loop
        bb_model.train()
        rp_model.train()
        for epoch in range(num_epochs):
            rationale_extractor.update_step(epoch)
            for batch in train_loader:

                if patience_left == 0:
                    pb.write("Patience is 0, early stopping")
                    break

                # generate prediction and token probs of being in a rationale
                batch.tokenized_examples = batch.tokenized_examples.to(device)
                att_pred, token_att = bb_model(**batch.tokenized_examples)

                with torch.no_grad():
                    hard_mask = rationale_selector(batch, token_att)
                    rationale, _, replace_ratio = rationale_extractor(batch, hard_mask)

                # predict from rationale
                hard_pred = rp_model(**rationale)

                bb_loss = bb_model.get_loss(
                    att_pred=att_pred,
                    hard_pred=hard_pred.detach(),
                    labels=batch.labels_bb.to(device),
                    proximity=proximity,
                )

                rp_loss = rp_model.get_loss(
                    att_pred=att_pred.detach(),
                    hard_pred=hard_pred,
                    labels=batch.labels_rp.to(device),
                    proximity=proximity,
                )

                bb_optimizer.zero_grad()
                rp_optimizer.zero_grad()

                bb_loss.backward()
                rp_loss.backward()

                bb_optimizer.step()
                rp_optimizer.step()

                pb.update(1)

                # update running values
                bb_running_train_loss += bb_loss.item()
                rp_running_train_loss += rp_loss.item()
                running_train_replace_ratio += replace_ratio
                global_step += 1

                # validation step
                if global_step % eval_every == 0:
                    bb_model.eval()
                    rp_model.eval()
                    with torch.no_grad():
                        for batch in valid_loader:
                            # generate prediction and token probs of being in a rationale
                            batch.tokenized_examples = batch.tokenized_examples.to(
                                device
                            )
                            att_pred, token_att = bb_model(**batch.tokenized_examples)

                            with torch.no_grad():
                                hard_mask = rationale_selector(batch, token_att)
                                rationale, _, replace_ratio = (
                                    validation_rationale_extractor(batch, hard_mask)
                                )

                            # predict from rationale
                            hard_pred = rp_model(**rationale)

                            bb_valid = bb_model.get_loss(
                                att_pred=att_pred,
                                hard_pred=hard_pred,
                                labels=batch.labels_bb.to(device),
                                proximity=proximity,
                            )

                            rp_valid = rp_model.get_loss(
                                att_pred=att_pred,
                                hard_pred=hard_pred,
                                labels=batch.labels_rp.to(device),
                                proximity=proximity,
                            )

                            bb_running_valid_loss += bb_valid.item()
                            rp_running_valid_loss += rp_valid.item()
                            running_valid_replace_ratio += replace_ratio

                    # evaluation
                    bb_average_train_loss = bb_running_train_loss / eval_every
                    rp_average_train_loss = rp_running_train_loss / eval_every
                    average_train_replace_ratio = (
                        running_train_replace_ratio / eval_every
                    )

                    bb_average_valid_loss = bb_running_valid_loss / len(valid_loader)
                    rp_average_valid_loss = rp_running_valid_loss / len(valid_loader)
                    average_valid_replace_ratio = running_valid_replace_ratio / len(
                        valid_loader
                    )

                    bb_improved = bb_best_valid_loss > bb_average_valid_loss
                    rp_improved = rp_best_valid_loss > rp_average_valid_loss

                    patience_left = (
                        patience if bb_improved and rp_improved else patience_left - 1
                    )

                    metrics.append(
                        {
                            "bb": {
                                "train_loss": bb_average_train_loss,
                                "valid_loss": bb_average_valid_loss,
                            },
                            "rp": {
                                "train_loss": rp_average_train_loss,
                                "valid_loss": rp_average_valid_loss,
                            },
                            "replace_ratio": {
                                "replace_train_ratio": average_train_replace_ratio,
                                "replace_valid_ratio": average_valid_replace_ratio,
                            },
                            "patience_left": patience_left,
                            "step": global_step,
                        }
                    )

                    # update running values
                    if bb_improved and rp_improved:
                        bb_best_train_loss = min(
                            bb_best_train_loss, bb_average_train_loss
                        )
                        bb_best_valid_loss = min(
                            bb_best_valid_loss, bb_average_valid_loss
                        )
                        rp_best_train_loss = min(
                            rp_best_train_loss, rp_average_train_loss
                        )
                        rp_best_valid_loss = min(
                            rp_best_valid_loss, rp_average_valid_loss
                        )

                    # resetting running values
                    bb_running_train_loss = 0.0
                    rp_running_train_loss = 0.0
                    running_train_replace_ratio = 0.0
                    bb_running_valid_loss = 0.0
                    rp_running_valid_loss = 0.0
                    running_valid_replace_ratio = 0.0

                    # print progress
                    pb.write(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}]"
                    )
                    pb.write(
                        f"Train Probability of Replacement: {average_train_replace_ratio * 100:.4f}"
                    )
                    pb.write(
                        f"Valid Probability of Replacement: {average_valid_replace_ratio * 100:.4f}"
                    )
                    pb.write(
                        f"BB Train Loss: {bb_average_train_loss:.4f}, BB Valid Loss: {bb_average_valid_loss:.4f}"
                    )
                    pb.write(
                        f"RP Train Loss: {rp_average_train_loss:.4f}, RP Valid Loss: {rp_average_valid_loss:.4f}"
                    )
                    pb.write(f"Patience: {patience_left}")

                    # checkpoint
                    if bb_improved and rp_improved:
                        pb.write(
                            f"Model saved to ==> {bb_model_save(bb_model, checkpoint_dir)}"
                        )
                        pb.write(
                            f"Model saved to ==> {rp_model_save(rp_model, checkpoint_dir)}"
                        )
                    pb.write(
                        f"Metrics saved to ==> {metrics_save(metrics, checkpoint_dir)}"
                    )

                    bb_model.train()
                    rp_model.train()

            if patience_left == 0:
                break


# In Progress
def evaluate(
    bb_model,
    rp_model,
    test_loader,
    device,
    show_detail,
    rationale_selector,
    rationale_extractor,
    sentence_selector,
    checkpoint_dir,
    result_path,
):

    if checkpoint_dir is not None:
        print("Loading BB model")
        bb_model_load(bb_model, checkpoint_dir)
        print("Loading RP model")
        rp_model_load(rp_model, checkpoint_dir)

    tp = 0
    fp = 0
    fn = 0

    rratio = 0.0

    rprec = 0
    rrec = 0
    rf1 = 0
    rtotal = 0

    y_pred = []
    y_true = []

    comp = []
    suff = []

    bb_model.eval()
    rp_model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch.tokenized_examples = batch.tokenized_examples.to(device)
            # generate prediction and token probs of being in a rationale
            _, token_att = bb_model(**batch.tokenized_examples)
            # mask based on probs
            hard_mask = rationale_selector(batch, token_att)
            # apply mask and recover rationale
            rationale, remainder, replace_ratio = rationale_extractor(batch, hard_mask)
            rratio += replace_ratio
            # predict from rationale
            hard_pred_logits = rp_model(**rationale)
            hard_pred_probs = torch.sigmoid(hard_pred_logits)

            y_pred.extend(torch.argmax(hard_pred_logits, 1).tolist())
            y_true.extend(batch.labels)

            label_pred_probs = get_label_pred_probs(hard_pred_probs, batch.labels)

            remainder_hard_pred_probs = torch.sigmoid(rp_model(**remainder))
            remainder_label_pred_probs = get_label_pred_probs(
                remainder_hard_pred_probs, batch.labels
            )

            all_hard_pred_probs = torch.sigmoid(rp_model(**batch.tokenized_examples))
            all_label_pred_probs = get_label_pred_probs(
                all_hard_pred_probs, batch.labels
            )

            comp.extend((all_label_pred_probs - remainder_label_pred_probs).tolist())
            suff.extend((all_label_pred_probs - label_pred_probs).tolist())

            selected_sentences = sentence_selector.get_selected_sentences(
                batch, token_att
            )
            for gen_sents, rat_sents in zip(selected_sentences, batch.rationale_ranges):
                if show_detail:
                    raise NotImplementedError(
                        "Show detail has not been implemented yet"
                    )

                gen_sents = set(gen_sents)
                rat_sents = set(rat_sents)

                rtp = len(gen_sents.intersection(rat_sents))
                tp += rtp
                rfn = len(rat_sents - gen_sents)
                fn += rfn
                rfp = len(gen_sents - rat_sents)
                fp += rfp

                rtotal += 1
                rprec += rtp / (rtp + rfp + 1e-6)
                rrec += rtp / (rtp + rfn + 1e-6)
                rf1 += rtp / (rtp + ((rfp + rfn) / 2) + 1e-6)

    results = {
        "rationales": {
            "micro": {
                "prec": tp / (tp + fp),
                "rec": tp / (tp + fn),
                "F1": tp / (tp + ((fp + fn) / 2)),
            },
            "macro": {"prec": rprec / rtotal, "rec": rrec / rtotal, "F1": rf1 / rtotal},
        },
        "sentence_selector_sparsity": sentence_selector.sparsity,
        "replace_ratio": rratio / len(test_loader),
        "comp_suff": {
            "comprehensiveness": sum(comp) / rtotal,
            "sufficiency": sum(suff) / rtotal,
        },
        "accuracy": classification_report(
            y_true, y_pred, labels=[1, 0], digits=4, output_dict=True
        )["accuracy"],
    }

    save_results(results, result_path)

    print("Rationales:")
    print(
        f"Sentence-level Micro-Averaged Precision: {tp/(tp + fp):.4f} Recall: {tp/(tp + fn):.4f} F1: {tp/(tp + ((fp + fn)/2)):.4f}"
    )
    print(
        f"Sentence-level Macro-Averaged Precision: {rprec/rtotal:.4f} Recall: {rrec/rtotal:.4f} F1: {rf1/rtotal:.4f}"
    )
    print(f"Replacement Ratio: {rratio/len(test_loader)}")
    print(f"Comprehensiveness: {sum(comp)/rtotal:.4f}")
    print(f"Sufficiency: {sum(suff)/rtotal:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))


def save_results(results, result_path):
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def get_label_pred_probs(pred_probs, labels):
    return torch.tensor(
        [pred_prob[label] for pred_prob, label in zip(pred_probs, labels)]
    )


def color_token(token, generated, handlabeled):
    if generated and handlabeled:
        return colored(token, "green")
    if not generated and handlabeled:
        return colored(token, "blue")
    if generated and not handlabeled:
        return colored(token, "red")
    return token


def model_save(model, path):
    torch.save(model.state_dict(), path)


def bb_model_save(model, path):
    save_path = os.path.join(path, "bb_model.pt")
    model_save(model, save_path)
    return save_path


def rp_model_save(model, path):
    save_path = os.path.join(path, "rp_model.pt")
    model_save(model, save_path)
    return save_path


def model_load(model, path):
    return model.load_state_dict(torch.load(path))


def bb_model_load(model, path):
    model_load(model, os.path.join(path, "bb_model.pt"))


def rp_model_load(model, path):
    model_load(model, os.path.join(path, "rp_model.pt"))


def metrics_save(metrics, path):
    save_path = os.path.join(path, "metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f)
    return save_path


if __name__ == "__main__":
    main(parse_args())
