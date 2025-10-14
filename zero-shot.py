import sar_clip
import argparse
import logging
import torch
from functools import partial
from contextlib import suppress
from tqdm import tqdm

# The class names used in zero-shot evaluation (semantic labels for different SAR image types)
CLASSNAMES = ("arable lands", "forests","grasslands", "open spaces or mineral", "urban areas", "water surfaces", "wetlands")

# Templates to generate textual descriptions from class names.
# Each lambda function receives a class name and returns a text prompt.
TEMPLATES = (
    lambda c: f'this is an SAR image of {c} .',
    lambda c: f'an SAR image shows {c} .',
    lambda c: f'the SAR image contains a scene with {c} .',
)


def parse_args():
    """
    Parse command-line arguments for the SARCLIP evaluation script.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for tokenizer file.",
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

def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy for model predictions.

    Args:
        output (Tensor): The model's output logits.
        target (Tensor): The ground-truth labels.
        topk (tuple): A tuple specifying which top-k accuracies to compute.

    Returns:
        List of accuracies for each k in topk.
    """
    # Get the indices of the top predictions for each sample.
    pred = output.topk(max(topk), 1, True, True)[1].t()
    # Compare the predictions with the target labels.
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # Calculate and return the number of correct predictions for each k.
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    """
    Runs the evaluation loop over the provided dataloader.

    Args:
        model: The SARCLIP model.
        classifier (Tensor): The zero-shot classifier embedding matrix.
        dataloader: A PyTorch DataLoader providing (images, target) pairs.
        args: Parsed command-line arguments.

    Returns:
        (top1, top3): Tuple containing the top-1 and top-3 accuracy scores.
    """
    device = torch.device(args.device)
    # Get the proper autocast context manager based on precision.
    autocast = get_autocast(args.precision, device_type=device.type)
    # Determine the input data type required by the model.
    input_dtype = sar_clip.get_input_dtype(args.precision)

    with torch.inference_mode(): # Disable gradient calculations.
        top1, top3, n = 0., 0., 0.
        # Iterate through the dataloader with a progress bar.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            # Move images and targets to the appropriate device and data type.
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            # Perform inference within the autocast context.
            with autocast():
                # Forward pass: get model output.
                output = model(image=images)
                # Extract image features from output.
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                # Compute logits by matrix multiplication with the classifier, scaled by 100.
                logits = 100. * image_features @ classifier

            # Compute accuracies (top-1 and top-3).
            acc1, acc3 = accuracy(logits, target, topk=(1, 3))
            top1 += acc1
            top3 += acc3
            n += images.size(0)

    # Calculate average accuracies over the dataset.
    top1 = (top1 / n)
    top3 = (top3 / n)
    return top1, top3

def zeroshot_evaluation(model, data, args, tokenizer):
    """
    Conducts zero-shot evaluation on the provided data using the SARCLIP model.

    Args:
        model: The SARCLIP model.
        data (dict): Dictionary containing dataset splits (e.g. 'imagenet-val').
        args: Parsed command-line arguments.
        tokenizer: Tokenizer used for text encoding.

    Returns:
        results (dict): Dictionary containing evaluation metrics.
    """
    # Check if the 'imagenet-val' dataset is provided.
    if 'imagenet-val' not in data:
        return {}

    logging.info('Starting zero-shot.')

    # If tokenizer is not provided, load it using SARCLIP helper.
    if tokenizer is None:
        tokenizer = sar_clip.get_tokenizer(args.model, cache_dir=args.cache_dir)

    logging.info('Building zero-shot classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    with autocast():
        # Build the zero-shot classifier using the model, tokenizer, class names, and templates.
        classifier = sar_clip.build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CLASSNAMES,
            templates=TEMPLATES,
            num_classes_per_batch=None,
            device=device,
            use_tqdm=True,
        )
    logging.info('Using classifier')
    results = {}
    # If the ImageNet validation data is available, evaluate the classifier.
    if 'imagenet-val' in data:
        top1, top3 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['zeroshot-val-top1'] = top1
        results['zeroshot-val-top3'] = top3

    logging.info('Finished zero-shot imagenet.')

    return results


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
    zero_shot_metrics = zeroshot_evaluation(model, data, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    # Print the evaluation metrics.
    print("========== Final Evaluation Metrics ==========")
    for metric, value in metrics.items():
        print(f"{metric:<35}: {value:.4f}")
    print("==============================================")


