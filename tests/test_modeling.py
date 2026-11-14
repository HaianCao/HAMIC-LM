from pathlib import Path

import pytest
import torch

from hamic.modeling import (
    Architecture,
    ModelFamily,
    ModelPreset,
    build_config,
    build_model,
    export_pretrained,
    parameter_report,
)
from hamic.tokenization import TokenizerKind
from tests.conftest import tiny_preset

PRESET_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"


@pytest.mark.parametrize("architecture", list(Architecture))
def test_every_architecture_builds_and_runs(architecture):
    preset = tiny_preset(architecture, vocab_size=300)
    model = build_model(preset, seed=0)
    ids = torch.randint(5, 300, (2, 12))
    out = model(input_ids=ids, labels=ids)
    assert out.logits.shape == (2, 12, 300)
    assert torch.isfinite(out.loss)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight


def test_qwen3_profile_has_qk_norm_and_llama_does_not():
    qwen = build_model(tiny_preset(Architecture.QWEN3, 300))
    llama = build_model(tiny_preset(Architecture.LLAMA, 300))
    assert hasattr(qwen.model.layers[0].self_attn, "q_norm")
    assert not hasattr(llama.model.layers[0].self_attn, "q_norm")


def test_modernbert_uses_full_attention_everywhere():
    config = build_config(tiny_preset(Architecture.MODERNBERT, 300))
    assert set(config.layer_types) == {"full_attention"}


def test_encoder_rejects_grouped_query_attention():
    with pytest.raises(ValueError, match="grouped-query"):
        ModelPreset(
            name="bad",
            architecture=Architecture.ROBERTA,
            label_parameters=1e5,
            num_layers=1,
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            vocab_size=100,
            max_position_embeddings=64,
        )


@pytest.mark.skipif(not PRESET_DIR.is_dir(), reason="model presets not present")
@pytest.mark.parametrize("path", sorted(PRESET_DIR.rglob("*.yaml")), ids=lambda p: p.stem)
def test_presets_match_their_name(path):
    report = parameter_report(ModelPreset.from_yaml(path))
    assert report.within_label, report.to_dict()


@pytest.mark.parametrize(
    ("architecture", "kind"),
    [
        (Architecture.MODERNBERT, TokenizerKind.ENCODER),
        (Architecture.QWEN3, TokenizerKind.DECODER),
    ],
)
def test_export_loads_with_plain_transformers(
    tmp_path, exported_tokenizers, tiny_tokenizer, architecture, kind
):
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, pipeline

    preset = tiny_preset(architecture, vocab_size=tiny_tokenizer.get_vocab_size())
    model = build_model(preset, seed=0)
    out = export_pretrained(model, preset, exported_tokenizers[kind], tmp_path / "export")

    if preset.family is ModelFamily.ENCODER:
        loaded = AutoModelForMaskedLM.from_pretrained(out)
        predictions = pipeline("fill-mask", model=str(out))("Thủ đô là Hà <mask>")
        assert len(predictions) == 5
    else:
        loaded = AutoModelForCausalLM.from_pretrained(out)
        generated = pipeline("text-generation", model=str(out))(
            "Thủ đô của Việt Nam là", max_new_tokens=3, do_sample=False
        )
        assert generated[0]["generated_text"].startswith("Thủ đô của Việt Nam là")
    assert type(loaded).__name__ == build_config(preset).architectures[0]
    with pytest.raises(FileExistsError):
        export_pretrained(model, preset, exported_tokenizers[kind], out)


ABLATION_PAIRS = ["e100m", "e300m", "d500m", "d1b"]


@pytest.mark.skipif(not (PRESET_DIR / "ablation").is_dir(), reason="ablation presets not present")
@pytest.mark.parametrize("family", ABLATION_PAIRS)
def test_ablation_presets_match_modern_parameters(family):
    ablation = parameter_report(
        ModelPreset.from_yaml(PRESET_DIR / f"ablation/{family}-ref-t1.yaml")
    )
    modern = parameter_report(ModelPreset.from_yaml(PRESET_DIR / f"hamic-{family}-m.yaml"))
    assert abs(ablation.total / modern.total - 1) < 0.01
    assert ModelPreset.from_yaml(PRESET_DIR / f"ablation/{family}-ref-t1.yaml").vocab_size == 20544
