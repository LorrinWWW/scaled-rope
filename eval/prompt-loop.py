import argparse
import sys

from datasets import load_dataset
from model_loader import apply_patches, load_model
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=sys.maxsize, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(
        args.model, args.load_in_8bit, args.load_in_4bit, args.max_new_tokens
    )
    apply_patches(
        model,
        args.max_new_tokens,
        args.dynamic_ntk,
        args.dynamic_linear,
        args.ntk,
        args.linear,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

    while True:
        if args.input_file is None:
            prompt_text = input("> ")
        else:
            input(f"Press enter to read {args.input_file} ")
            prompt_text = open(args.input_file, encoding="utf=8").read()
        response = pipe(
            prompt_text, num_return_sequences=1, max_new_tokens=args.max_new_tokens
        )[0]["generated_text"][len(prompt_text) :]
        print(f"< {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--linear", type=float)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--input-file", type=str)

    args = parser.parse_args()
    main(args)
