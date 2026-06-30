"""
Upload a locally fine-tuned transformer to the Hugging Face Hub so the deployed
app (e.g. Streamlit Cloud) can load weights that are too large to commit to git.

Why: fine-tuned FinBERT weights are ~417 MB — over GitHub's 100 MB/file limit.
Hosting them on the HF Hub lets the app download them at runtime when local
files are absent (see src/predict.py::_finetuned_hf_repo).

Usage
-----
1. Create a free account at https://huggingface.co and a token (write scope):
   https://huggingface.co/settings/tokens
2. Authenticate once:
       huggingface-cli login          # paste your token
   (or set the HF_TOKEN environment variable)
3. Upload the fine-tuned FinBERT:
       python scripts/upload_finetuned_to_hf.py --model finbert --repo your-username/finbert-financial
4. On Streamlit Cloud, add an app secret:
       FINSIGHT_FINBERT_REPO = "your-username/finbert-financial"
   The model then appears in the selector and loads from the Hub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import get_model_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a fine-tuned model to the Hugging Face Hub")
    parser.add_argument("--model", default="finbert", help="Model name (matches models/<model>_finetuned/)")
    parser.add_argument("--repo", required=True, help="Target HF repo id, e.g. your-username/finbert-financial")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--token", default=None, help="HF token (defaults to cached login / HF_TOKEN env var)")
    args = parser.parse_args()

    local_dir = get_model_dir() / f"{args.model}_finetuned"
    if not local_dir.exists():
        raise SystemExit(f"[error] Local weights not found: {local_dir}")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("[error] huggingface_hub not installed. Run: pip install huggingface_hub") from exc

    print(f"[info] Creating/ensuring repo: {args.repo} (private={args.private})")
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True, token=args.token)

    print(f"[info] Uploading {local_dir} -> {args.repo} ...")
    api = HfApi(token=args.token)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"Upload fine-tuned {args.model}",
    )

    print("\n[ok] Upload complete.")
    print(f"     Model page: https://huggingface.co/{args.repo}")
    print(f'     Set this secret on your deployment:  FINSIGHT_{args.model.upper()}_REPO = "{args.repo}"')


if __name__ == "__main__":
    main()
