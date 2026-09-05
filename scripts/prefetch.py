#!/usr/bin/env python3
"""Download models and datasets into the HF cache. **Run on the login node.**

Compute nodes on this cluster have no outbound internet; the login node does.
A job that tries to reach huggingface.co dies after a long timeout with a
`LocalEntryNotFoundError` wrapped in an `OSError` about the connection, which
reads like a network blip rather than a policy. So everything a job needs must
be in `~/.cache/huggingface` before the job starts.

    # on the login node, NOT under srun
    uv run --no-sync python scripts/prefetch.py \
      --model EleutherAI/pythia-410m --dataset wikitext:wikitext-2-raw-v1

    # VLMs (LLaVA, Qwen-VL, ...) need --model-class so the processor -- not
    # just a text tokenizer -- gets cached:
    uv run --no-sync python scripts/prefetch.py \
      --model llava-hf/llava-1.5-7b-hf --model-class image-text-to-text

Then run jobs with `HF_HUB_OFFLINE=1` so a cache miss fails immediately and says
what is missing, instead of hanging on a connection that cannot succeed.
`slurm/run.slurm` exports it for you.

The two guards below exist because the mistake is easy and the symptom is
misleading: this refuses to run under Slurm, and refuses to run from a host that
cannot reach the hub.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys


def _has_internet(host: str = "huggingface.co", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", action="append", default=[],
                   help="HF repo id; repeatable")
    p.add_argument("--revision", default=None,
                   help="pin a revision; applies to every --model")
    p.add_argument("--dataset", action="append", default=[],
                   help="HF dataset as 'name' or 'name:config'; repeatable")
    p.add_argument("--tokenizer-only", action="store_true",
                   help="skip the weights (much faster when you only need configs)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--model-class", default="causal-lm",
                   choices=["causal-lm", "image-text-to-text"],
                   help="architecture family for every --model in this invocation "
                        "(default: causal-lm, i.e. plain text LLMs). Use "
                        "image-text-to-text for VLMs (LLaVA, Qwen-VL, ...) -- it "
                        "prefetches an AutoProcessor (image processor + tokenizer) "
                        "instead of a text-only AutoTokenizer.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model and not args.dataset:
        print("nothing to do: pass --model and/or --dataset", file=sys.stderr)
        return 2

    if os.environ.get("SLURM_JOB_ID"):
        print("REFUSING: this is running under Slurm. Compute nodes have no internet;\n"
              "          run it on the login node instead.", file=sys.stderr)
        return 2

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("REFUSING: HF_HUB_OFFLINE=1 is set, so nothing would be downloaded.\n"
              "          Unset it for the prefetch; jobs keep it set.", file=sys.stderr)
        return 2

    if not _has_internet():
        print("REFUSING: cannot reach huggingface.co:443 from this host.\n"
              "          Prefetch must run on the login node.", file=sys.stderr)
        return 2

    failures = []

    for name in args.model:
        print(f"=== model: {name} ({args.model_class}) ===", flush=True)
        try:
            from transformers import AutoConfig

            kwargs = {"revision": args.revision, "trust_remote_code": args.trust_remote_code}
            config = AutoConfig.from_pretrained(name, **kwargs)

            if args.model_class == "image-text-to-text":
                from transformers import AutoModelForImageTextToText, AutoProcessor
                # AutoProcessor bundles the image processor + tokenizer a VLM
                # needs; a plain AutoTokenizer would silently drop the image side.
                AutoProcessor.from_pretrained(name, **kwargs)
                model_cls = AutoModelForImageTextToText
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                AutoTokenizer.from_pretrained(name, **kwargs)
                model_cls = AutoModelForCausalLM

            if not args.tokenizer_only:
                # Materialize the weights so the shards land in the cache.
                # Loading to CPU is enough -- the point is the download, not the
                # device, and the login node's GPU is not yours to occupy.
                model_cls.from_pretrained(name, **kwargs)
            print(f"  ok: {config.model_type}, "
                  f"{getattr(config, 'num_hidden_layers', '?')} layers", flush=True)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    for spec in args.dataset:
        path, _, config_name = spec.partition(":")
        print(f"=== dataset: {spec} ===", flush=True)
        try:
            from datasets import load_dataset

            ds = load_dataset(path, config_name or None)
            for split, part in ds.items():
                print(f"  ok: {split}, {part.num_rows:,} rows", flush=True)
        except Exception as exc:
            failures.append(f"{spec}: {type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"\n{len(failures)} failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll cached. Run jobs with HF_HUB_OFFLINE=1 so a cache miss fails fast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
