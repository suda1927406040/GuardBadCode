#!/usr/bin/env python
"""GuardBadCode — CodeT5 code-summarization backdoor pipeline (single entry).

Full flow, three stages (layout follows EliBadCode):

    python run.py poison --train_jsonl clean.jsonl --output_dir data/cs_testo
    python run.py train  --train_filename data/cs_testo/train_poisoned.jsonl ...
    python run.py defense --checkpoint outputs/codet5_cs_train/checkpoint-best-bleu/pytorch_model.bin ...

Each command forwards all extra CLI arguments to its stage script
(`python run.py <cmd> --help` shows the stage's own options).
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGES = {
    # command -> (file, module name)
    "poison": (REPO_ROOT / "poison.py", "poison"),
    "train": (REPO_ROOT / "attacks/codet5_summarization/train.py", "codet5_train"),
    "defense": (REPO_ROOT / "defense/run.py", "defense_run"),
}

USAGE = "\n".join(f"  {cmd:<10} -> {path.relative_to(REPO_ROOT)}" for cmd, (path, _) in STAGES.items())


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.rstrip())
        print("\ncommand -> stage:")
        print(USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd not in STAGES:
        print(f"error: unknown command '{cmd}'\n", file=sys.stderr)
        print(__doc__.rstrip())
        print("\ncommand -> stage:")
        print(USAGE)
        return 2

    path, modname = STAGES[cmd]
    spec = importlib.util.spec_from_file_location(modname, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    sys.argv = [str(path)] + rest
    spec.loader.exec_module(module)  # loads stage definitions
    entry = getattr(module, "main", None)
    if entry is not None:  # stages guard their CLI with if __name__ == "__main__"
        entry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
