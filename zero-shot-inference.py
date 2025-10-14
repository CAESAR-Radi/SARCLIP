import sar_clip
import argparse
import logging
import torch
from functools import partial
from contextlib import suppress
from tqdm import tqdm
import os
import glob
import csv
from torch.utils.data import Dataset

# The class names used in zero-shot evaluation (semantic labels for different SAR image types).
CLASSNAMES = (
    "arable lands", "forests", "grasslands", "open spaces or mineral",
    "urban areas", "water surfaces", "wetlands"
)

# Templates to generate textual descriptions from class names.
# Each lambda function receives a class name and returns a text prompt.
TEMPLATES = (
    lambda c: f'this is an SAR image of {c} .',
    lambda c: f'an SAR image shows {c} .',
    lambda c: f'the SAR image contains a scene with {c} .',
)

def parse_args():
    """
    Parse command-line arguments for the SARCLIP inference script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for tokenizer file."
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
        help="Path to a pretrained SARCLIP model weights file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use."
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
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to the folder containing images to be classified (unlabeled data)."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="classification_results.csv",
        help="CSV file path to store classification results."
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Unknown args]: {unknown}")
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
        return suppress

    return partial(torch.amp.autocast, device_type=device_type, dtype=amp_dtype)

class UnlabeledImageFolder(Dataset):
    """
    Custom dataset for reading images from a specified folder (unlabeled).
    Returns the image and the corresponding file path for later output.
    """
    def __init__(self, root, transform=None, extensions=("*.tif", "*.tiff")):
        self.root = root
        self.transform = transform
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(root, ext)))
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = sar_clip.readtif(image_path)
        if self.transform:
            image = self.transform(image)
        return image, image_path

def run(model, classifier, dataloader, args):
    """
    Performs zero-shot inference on an unlabeled dataset.

    Args:
        model: The SARCLIP model.
        classifier (Tensor): The zero-shot classifier embedding matrix.
        dataloader: DataLoader providing (images, file path) pairs.
        args: Parsed command-line arguments.

    Returns:
        List of dictionaries. Each dictionary contains:
            - 'filename': Image file path.
            - 'predicted_class': Predicted class name.
            - 'score': Confidence score.
    """
    device = torch.device(args.device)
    autocast_context = get_autocast(args.precision, device_type=device.type)
    input_dtype = sar_clip.get_input_dtype(args.precision)
    model.eval()

    results = []
    with torch.inference_mode():
        for images, filenames in tqdm(dataloader, desc="Classifying images", unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            with autocast_context():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier
            probs = logits.softmax(dim=1)
            top1 = probs.argmax(dim=1)
            for i, fn in enumerate(filenames):
                pred_class = CLASSNAMES[top1[i].item()]
                score = probs[i, top1[i]].item()
                results.append({
                    'filename': fn,
                    'predicted_class': pred_class,
                    'score': score
                })
    return results

def zeroshot_inference(model, args):
    """
    Conducts zero-shot inference on a folder of unlabeled images.

    Args:
        model: The SARCLIP model.
        args: Parsed command-line arguments.

    Returns:
        List of dictionaries with inference results (filename, predicted class, score).
    """
    # Build the zero-shot classifier (used for inference).
    logging.info("Building zero-shot classifier.")
    device = torch.device(args.device)
    autocast_context = get_autocast(args.precision, device_type=device.type)
    with autocast_context():
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

    dataset = UnlabeledImageFolder(args.image_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    results = run(model, classifier, dataloader, args)
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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

    if args.image_dir is not None:
        results = zeroshot_inference(model, args)
        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['filename', 'predicted_class', 'score'])
            writer.writeheader()
            writer.writerows(results)
        logging.info("Classification results saved to %s", args.output_csv)
        print(f"Classification results saved to {args.output_csv}")
    else:
        logging.info("No image directory specified. Exiting.")
