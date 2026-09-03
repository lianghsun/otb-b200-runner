#!/usr/bin/env python3
"""Strip the vision tower off MediaTek's Llama-Breeze2-3B-Instruct so vLLM can
serve its language model.

Breeze2 is an InternVL wrapper (`InternVLChatModel`): an InternViT-6B vision
encoder + a Llama-3.2-3B language model, glued by an `mlp1` projector. vLLM has
no loader for that custom multimodal class, but the LLM inside is a plain
`LlamaForCausalLM`. This copies out only the `language_model.*` weights (with
the prefix removed), writes a standard Llama config + tokenizer, and saves a
text-only checkpoint that vLLM loads like any other Llama.

    HF_TOKEN=hf_xxx python prepare_breeze2.py           # -> ./breeze2-3b-text

eval_vllm.py maps "MediaTek-Research/Llama-Breeze2-3B-Instruct" to this local
dir (see LOCAL_ALIAS), so the board still shows the real model name.
"""
import json
import os
import pathlib
import shutil

SRC = "MediaTek-Research/Llama-Breeze2-3B-Instruct"
OUT = pathlib.Path(__file__).resolve().parent / "breeze2-3b-text"


def main():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    print(f"== downloading {SRC}", flush=True)
    src = pathlib.Path(snapshot_download(SRC, token=os.environ.get("HF_TOKEN")))

    cfg = json.loads((src / "config.json").read_text())
    llm = cfg.get("llm_config") or cfg.get("text_config")
    if not llm:
        raise SystemExit("no llm_config/text_config in config.json — check the model")
    llm.setdefault("architectures", ["LlamaForCausalLM"])
    llm["model_type"] = "llama"
    # carry a couple of top-level bits the LLM config may rely on
    if "torch_dtype" not in llm and "torch_dtype" in cfg:
        llm["torch_dtype"] = cfg["torch_dtype"]

    OUT.mkdir(parents=True, exist_ok=True)

    # --- collect the safetensors shards ---
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit("no .safetensors in the snapshot")
    kept, dropped = {}, 0
    for shard in shards:
        sd = load_file(str(shard))
        for k, v in sd.items():
            if k.startswith("language_model."):
                kept[k[len("language_model."):]] = v   # -> model.* / lm_head.*
            else:
                dropped += 1
    if not kept:
        raise SystemExit("no 'language_model.*' tensors found — key prefix differs; "
                         "inspect the checkpoint keys and adjust the prefix")
    print(f"   kept {len(kept)} LM tensors, dropped {dropped} vision/projector tensors",
          flush=True)

    # a tied-embedding Llama-3.2 has no separate lm_head; that's fine, vLLM ties it
    save_file(kept, str(OUT / "model.safetensors"), metadata={"format": "pt"})
    (OUT / "config.json").write_text(json.dumps(llm, ensure_ascii=False, indent=2))

    # --- tokenizer + generation config (text side is a normal Llama-3.2 tokenizer) ---
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "tokenizer.model", "generation_config.json", "chat_template.jinja",
                 "chat_template.json"):
        p = src / name
        if p.exists():
            shutil.copy2(p, OUT / name)

    print(f"== done -> {OUT}\n   run:  MODELS='{OUT}' bash bootstrap.sh   (or it's in DEFAULT_MODELS via LOCAL_ALIAS)",
          flush=True)


if __name__ == "__main__":
    main()
