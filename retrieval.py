import sar_clip
import argparse
import logging
import torch
from functools import partial
from contextlib import suppress
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def parse_args():
    """
    Parse command-line arguments for the SARCLIP evaluation script.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for tokenizer file.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--dataset-type",
        default="csv",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )

    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size per GPU."
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained SARCLIP model weights with the file path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )

    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(f'[Unknow args]: {unknown}')
    return args

def get_autocast(precision, device_type='cuda'):
    """
    Returns an automatic mixed-precision (AMP) context manager based on the precision setting.

    Args:
        precision (str): The precision mode to use (e.g., 'amp', 'fp32').
        device_type (str): The device type (default 'cuda').

    Returns:
        A context manager that enables AMP if applicable, otherwise a no-op context manager using suppress.
    """
    if precision =='amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        amp_dtype = torch.bfloat16
    else:
        # If precision is not one of the AMP options, return a no-op context manager using suppress.
        return suppress

    # Return a partially-applied torch.amp.autocast function with the desired device and dtype.
    return partial(torch.amp.autocast, device_type=device_type, dtype=amp_dtype)



def retrieval(model, data, args):
    logging.info('Starting retrieval')

    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = sar_clip.get_input_dtype(args.precision)

    if 'val' in data:
        dataloader = data['val'].dataloader

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for images, texts in tqdm(dataloader, unit_scale=args.batch_size):
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )

    return {**val_metrics}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Parse command-line arguments.
    args = parse_args()

    # Set up the model.
    model = sar_clip.create_model_with_args(
        args.model,
        pretrained=args.pretrained,
        precision=args.precision,
        device=args.device,
        cache_dir=args.cache_dir,
        output_dict=True
    )

    # Load the tokenizer.
    tokenizer = sar_clip.get_tokenizer(args.model, cache_dir=args.cache_dir)

    # Retrieve data required for evaluation.
    data = sar_clip.get_data(
        args,
        tokenizer=tokenizer,
    )

    metrics = {}

    device = torch.device(args.device)
    model.eval() # Set model to evaluation mode.

    # Perform zero-shot evaluation.
    zero_shot_metrics = retrieval(model, data, args)
    metrics.update(zero_shot_metrics)

    # Print the evaluation metrics.
    print("========== Final Evaluation Metrics ==========")
    for metric, value in metrics.items():
        print(f"{metric:<35}: {value:.4f}")
    print("==============================================")


