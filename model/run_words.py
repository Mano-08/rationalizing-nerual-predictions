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
    SelectorFactory,
)
from movies import DataLoaderFactory

logging.set_verbosity_error()


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
    parser.add_argument("--num_epochs", type=int, default=20)
    # Patience
    parser.add_argument("--patience", type=int, default=2)
    # Model proximity hyperparameter
    parser.add_argument("--proximity", type=float, default=0.1)
    # Model
    parser.add_argument(
        "--save_path", type=str, default=os.path.join("trained", "ours")
    )
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    # Rationale Extraction hyperparameter
    parser.add_argument("--sparsity", type=float, default=0.2)
    # Dataset
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("..", "..", "rnp_movie_review", "original"),
    )
    # Selection method
    parser.add_argument(
        "--selection_method", choices=["words", "span"], default="words"
    )
    # Eval-related
    # Compare model-generated and hand-labeled rationales
    parser.add_argument("--show_detail", action="store_true")
    return parser.parse_args()


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
        num_labels=3, model=args.model, freeze_encoder=args.freeze_encoder_rp
    ).to(args.device)
    print(f"Rationale Predictor: {get_num_params(rp_model)} parameters")

    rationale_selector = SelectorFactory(
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
            eval_every=len(train_loader),
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
        ).create_dataloader("test", False)

        test_rationale_extractor = RationaleExtractor(tokenizer, args.device)

        evaluate(
            bb_model=bb_model,
            rp_model=rp_model,
            tokenizer=tokenizer,
            test_loader=test_loader,
            device=args.device,
            show_detail=args.show_detail,
            rationale_selector=rationale_selector,
            rationale_extractor=test_rationale_extractor,
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
                batch.reviews_tokenized = batch.reviews_tokenized.to(device)
                att_pred, token_att = bb_model(**batch.reviews_tokenized)

                hard_mask = rationale_selector(
                    token_att=token_att, input_ids=batch.reviews_tokenized.input_ids
                )
                rationale, _, replace_ratio = rationale_extractor(
                    batch=batch, hard_mask=hard_mask
                )

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
                            batch.reviews_tokenized = batch.reviews_tokenized.to(device)
                            att_pred, token_att = bb_model(**batch.reviews_tokenized)

                            hard_mask = rationale_selector(
                                token_att=token_att,
                                input_ids=batch.reviews_tokenized.input_ids,
                            )

                            rationale, _, replace_ratio = (
                                validation_rationale_extractor(
                                    batch=batch, hard_mask=hard_mask
                                )
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

                    # bb_improved = bb_best_train_loss > bb_average_train_loss and bb_best_valid_loss > bb_average_valid_loss
                    # rp_improved = rp_best_train_loss > rp_average_train_loss and rp_best_valid_loss > rp_average_valid_loss
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
                    bb_best_train_loss = min(bb_best_train_loss, bb_average_train_loss)
                    bb_best_valid_loss = min(bb_best_valid_loss, bb_average_valid_loss)
                    rp_best_train_loss = min(rp_best_train_loss, rp_average_train_loss)
                    rp_best_valid_loss = min(rp_best_valid_loss, rp_average_valid_loss)

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


def evaluate(
    bb_model,
    rp_model,
    tokenizer,
    test_loader,
    device,
    show_detail,
    rationale_selector,
    rationale_extractor,
    checkpoint_dir,
    result_path,
):

    if checkpoint_dir is not None:
        print("Loading BB model")
        bb_model_load(bb_model, checkpoint_dir)
        print("Loading RP model")
        rp_model_load(rp_model, checkpoint_dir)

    if show_detail:
        detail_path = os.path.join(os.path.dirname(checkpoint_dir), "details")
        os.makedirs(detail_path, exist_ok=True)

    gen_spans = 0
    rat_spans = 0
    gen_rat_span_ratio = 0.0
    gen_rat_span_rtotal = 0
    max_gen_span = torch.zeros(1)
    max_rat_span = torch.zeros(1)

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

    ious = []
    num_gen_tokens = []
    num_rat_tokens = []

    review_count = 0

    bb_model.eval()
    rp_model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch.reviews_tokenized = batch.reviews_tokenized.to(device)
            # generate prediction and token probs of being in a rationale
            _, token_att = bb_model(**batch.reviews_tokenized)
            # mask based on probs
            hard_mask = rationale_selector(
                token_att=token_att, input_ids=batch.reviews_tokenized.input_ids
            )
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

            all_hard_pred_probs = torch.sigmoid(rp_model(**batch.reviews_tokenized))
            all_label_pred_probs = get_label_pred_probs(
                all_hard_pred_probs, batch.labels
            )

            comp.extend((all_label_pred_probs - remainder_label_pred_probs).tolist())
            suff.extend((all_label_pred_probs - label_pred_probs).tolist())

            for i in range(hard_mask.shape[0]):
                gen_mask = torch.tensor([False] + hard_mask[i, :, :].squeeze().tolist())
                rat_mask = torch.tensor(
                    [
                        any(
                            id in range(low, high)
                            for (low, high) in batch.rationale_ranges[i]
                        )
                        for id in batch.reviews_tokenized.word_ids(i)
                    ]
                )

                gen_span = torch.logical_and(
                    gen_mask[:-1] == False, gen_mask[1:] == True
                ).sum()
                gen_spans += gen_span
                max_gen_span = torch.max(
                    torch.stack([max_gen_span.squeeze(), gen_span.squeeze()])
                )
                rat_span = torch.logical_and(
                    rat_mask[:-1] == False, rat_mask[1:] == True
                ).sum()
                max_rat_span = torch.max(
                    torch.stack([max_rat_span.squeeze(), rat_span.squeeze()])
                )
                rat_spans += rat_span
                if not rat_span == torch.zeros(
                    1
                ):  # rat_span is LongTensor, so it works
                    gen_rat_span_ratio += gen_span / (rat_span)
                    gen_rat_span_rtotal += 1

                rtp = torch.sum(gen_mask & rat_mask).item()
                tp += rtp
                rfn = torch.sum(~gen_mask & rat_mask).item()
                fn += rfn
                rfp = torch.sum(gen_mask & ~rat_mask).item()
                fp += rfp

                rtotal += 1
                rprec += rtp / (rtp + rfp + 1e-6)
                rrec += rtp / (rtp + rfn + 1e-6)
                rf1 += rtp / (rtp + ((rfp + rfn) / 2) + 1e-6)

                if show_detail:
                    with open(
                        os.path.join(detail_path, f"review_{review_count}.txt"), "w"
                    ) as f:
                        review_tokens = tokenizer.convert_ids_to_tokens(
                            batch.reviews_tokenized.input_ids[i]
                        )
                        review_tokens_colored = [
                            color_token(token, g, h)
                            for token, g, h in zip(review_tokens, gen_mask, rat_mask)
                        ]
                        print(" ".join(review_tokens_colored), file=f)
                        print(
                            f"Class: {'POSITIVE' if batch.labels[i] else 'NEGATIVE'}",
                            file=f,
                        )
                        print(
                            f"P: {100*rtp/(rtp + rfp + 1e-6):.2f} R: {100*rtp/(rtp + rfn + 1e-6):.2f} F1: {100*rtp/(rtp + ((rfp + rfn)/2) + 1e-6):.2f}",
                            file=f,
                        )
                    review_count += 1

                gen_sets = to_sets(to_ranges(gen_mask))
                num_gen_tokens.append(sum([len(s) for s in gen_sets]))
                rat_sets = to_sets(to_ranges(rat_mask))
                num_rat_tokens.append(sum([len(s) for s in rat_sets]))

                rious = [
                    (
                        max(
                            [
                                len(gen_set & rat_set) / (len(gen_set | rat_set) + 1e-6)
                                for rat_set in rat_sets
                            ]
                            + [0.0]
                        ),
                        len(gen_set),
                    )
                    for gen_set in gen_sets
                ]
                ious.append(rious)

    micro_prec = tp / (tp + fp)
    micro_rec = tp / (tp + fn)
    micro_f1 = tp / (tp + ((fp + fn) / 2))
    macro_prec = rprec / rtotal
    macro_rec = rrec / rtotal
    macro_f1 = rf1 / rtotal

    micro_iou = dict()
    macro_iou = dict()

    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for threshold in iou_thresholds:
        thresholded_ious = [
            sum([int(riou >= threshold) * riou_tokens for riou, riou_tokens in rious])
            for rious in ious
        ]

        micro_iou[threshold] = dict()
        micro_iou[threshold]["prec"] = sum(thresholded_ious) / sum(num_gen_tokens)
        micro_iou[threshold]["rec"] = sum(thresholded_ious) / sum(num_rat_tokens)
        micro_iou[threshold]["f1"] = (
            2 * micro_iou[threshold]["prec"] * micro_iou[threshold]["rec"]
        ) / (micro_iou[threshold]["prec"] + micro_iou[threshold]["rec"])

        iou_rprec = [x / (y + 1e-6) for x, y in zip(thresholded_ious, num_gen_tokens)]
        iou_rrec = [x / (y + 1e-6) for x, y in zip(thresholded_ious, num_rat_tokens)]
        macro_iou[threshold] = dict()
        macro_iou[threshold]["prec"] = sum(iou_rprec) / len(iou_rprec)
        macro_iou[threshold]["rec"] = sum(iou_rrec) / len(iou_rrec)
        macro_iou[threshold]["f1"] = (
            2 * macro_iou[threshold]["prec"] * macro_iou[threshold]["rec"]
        ) / (macro_iou[threshold]["prec"] + macro_iou[threshold]["rec"])

    results = {
        "rationales": {
            "micro": {"prec": micro_prec, "rec": micro_rec, "F1": micro_f1},
            "macro": {"prec": macro_prec, "rec": macro_rec, "F1": macro_f1},
        },
        "token_selector_sparsity": rationale_selector.sparsity,
        "replace_ratio": rratio / len(test_loader),
        "comp_suff": {
            "comprehensiveness": sum(comp) / rtotal,
            "sufficiency": sum(suff) / rtotal,
        },
        "macro_iou": macro_iou,
        "micro_iou": micro_iou,
        "accuracy": classification_report(
            y_true, y_pred, labels=[2, 1, 0], digits=4, output_dict=True
        )["accuracy"],
    }

    if not show_detail:
        save_results(results, result_path)

    print("Rationales:")
    print(
        f"Token-level Micro-Averaged Precision: {micro_prec:.4f} Recall: {micro_rec:.4f} F1: {micro_f1:.4f}"
    )
    print(
        f"Token-level Macro-Averaged Precision: {macro_prec:.4f} Recall: {macro_rec:.4f} F1: {macro_f1:.4f}"
    )
    for t in iou_thresholds:
        print(
            f"Token-level IOU Micro-Averaged Precision (threshold={t}): {micro_iou[t]['prec']:.4f} Recall: {micro_iou[t]['rec']:.4f} F1: {micro_iou[t]['f1']:.4f}"
        )
        print(
            f"Token-level IOU Macro-Averaged Precision (threshold={t}): {macro_iou[t]['prec']:.4f} Recall: {macro_iou[t]['rec']:.4f} F1: {macro_iou[t]['f1']:.4f}"
        )
    print(f"Replacement Ratio: {rratio/len(test_loader)}")
    print(
        f"Average number of generated spans: {gen_spans/rtotal:.4f}, labeled rationale spans: {rat_spans/rtotal:.4f}"
    )
    print(
        f"Maximum number of generated spans: {max_gen_span:.0f}, labeled rationale spans: {max_rat_span:.0f}"
    )
    print(
        f"Average ratio of generated spans to labeled rationale spans: {gen_rat_span_ratio/gen_rat_span_rtotal:.4f}"
    )
    print(f"Comprehensiveness: {sum(comp)/rtotal:.4f}")
    print(f"Sufficiency: {sum(suff)/rtotal:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=[2, 1, 0], digits=4))


def save_results(results, result_path):
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)


def to_ranges(mask):
    t1 = torch.tensor(mask.tolist() + [0], dtype=torch.bool)
    t2 = torch.tensor([0] + mask.tolist(), dtype=torch.bool)
    start = torch.logical_and(t1, ~t2)
    end = torch.logical_and(~t1, t2)
    indices = torch.arange(len(t1))
    return list(zip(indices[start].tolist(), indices[end].tolist()))


def to_sets(ranges):
    return [set(range(low, high)) for low, high in ranges]


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
