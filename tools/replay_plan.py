"""Offline replay of one captured crystal-bow plan: no model calls at all."""
import json
import sys
from dataclasses import replace
from pathlib import Path

from studio_next.contracts import (
    AppearanceSpec, AssetForm, AssetRequest, GeometrySpec, PartSpec, PrimitiveSpec,
    ReferenceAsset, ReferenceRole, ShapeDescriptor, appearance_from_dict,
)
from studio_next.pipeline import GenerationPipeline
from studio_next.quality import remap_appearance_to_anchor, sprite_palette_ramp
from studio_next.pipeline import _reference_shape_is_locked

ROOT = Path("/home/gmh/mc-art")
SRC = ROOT / sys.argv[1]
OUT = Path(sys.argv[2])


def captured(name):
    return json.loads((SRC / "llm_raw" / name).read_text(encoding="utf-8"))


# Reference name -> blob path, recovered from an earlier run of the same family.
ref_names = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
names = {e["name"]: e["path"] for e in ref_names}

ROUND = sys.argv[7] if len(sys.argv) > 7 else "round_00"


def artifact(name, fallback):
    """Prefer the normalized plan the run actually used over a raw response."""
    path = SRC / ROUND / "generated" / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return captured(fallback)


raw_desc = artifact("shape_descriptor.json", "002.response.txt")
graw = artifact("geometry.json", "003.response.txt")
araw = artifact("appearance.json", "004.response.txt")

parts = [PartSpec(
    id=p["id"], meaning=p["meaning"], required=p["required"], paint_only=p["paint_only"],
    layer=p["layer"], style_role=p.get("style_role", ""), contour_intent=p.get("contour_intent", "free"),
) for p in raw_desc["parts"]]
descriptor = ShapeDescriptor(
    target=raw_desc["target"], semantic=raw_desc["semantic"],
    visual_identity=raw_desc["visual_identity"], parts=parts,
    negative_identities=raw_desc.get("negative_identities", []),
    orientation=raw_desc.get("orientation", "auto"),
    reference_strategy=raw_desc.get("reference_strategy", ""),
    shape_edit_mode=raw_desc.get("shape_edit_mode", "model_decides"),
    reference_part_map=raw_desc.get("reference_part_map"),
)
geometry = GeometrySpec(
    width=graw["width"], height=graw["height"], parts=parts,
    primitives=[PrimitiveSpec(id=p["id"], part_id=p["part_id"], primitive=p["primitive"],
                              params=p.get("params", {}), layer=p.get("layer", 0))
                for p in graw["primitives"]],
)
FIELDS = {"colors", "material", "shade_axis", "noise", "highlight_ratio", "marks"}
sanitised = dict(araw)
if isinstance(sanitised.get("parts"), dict):
    sanitised["parts"] = {
        key: ({k: v for k, v in value.items() if k in FIELDS} if isinstance(value, dict) else value)
        for key, value in sanitised["parts"].items()
    }
appearance = appearance_from_dict(sanitised)
references = [
    ReferenceAsset(path=str(ROOT / path), name=name, roles=[ReferenceRole.SHAPE, ReferenceRole.PIXEL_STYLE])
    for name, path in names.items() if name.startswith("bow:")
]
references += [
    ReferenceAsset(path=str(ROOT / path), name=name, roles=[ReferenceRole.PIXEL_STYLE, ReferenceRole.PALETTE])
    for name, path in names.items() if not name.startswith("bow:")
]

if len(sys.argv) > 5 and sys.argv[5]:
    # Simulate the family-level inheritance: frames of one object share the
    # anchor's reading of whether the source contour is the answer.
    descriptor = replace(descriptor, shape_edit_mode=sys.argv[5])

if len(sys.argv) > 6 and sys.argv[6]:
    # Simulate the family palette contract: the model keeps its swatch names
    # and their dark/mid/light order, the family owns the values.
    ramp = sprite_palette_ramp(sys.argv[6])
    appearance = remap_appearance_to_anchor(appearance, ramp)

request = AssetRequest(query=descriptor.target, form=AssetForm.ITEM, name=sys.argv[4], width=16, height=16)
print("locked:", _reference_shape_is_locked(descriptor))

from studio_next.plans import GenerationPlan
plan = GenerationPlan(request=request, descriptor=descriptor, geometry=geometry,
                      appearance=appearance, references=references)


class NoModel:
    """Every model port, stubbed: the replay must be deterministic and free."""

    repair_soft_failures = False

    def review(self, descriptor, compiled):
        from studio_next.validation import ValidationResult
        return ValidationResult(passed=True, stage="offline_replay", metrics={}, warnings=[])

    def repair_geometry(self, *a, **k):
        raise AssertionError("offline replay must not call the model")


pipeline = GenerationPipeline(critic=NoModel(), repairer=NoModel(), max_geometry_repairs=0)
result = pipeline.run(plan, OUT, package=False)
print("sprite:", result.sprite_path)
print("validation passed:", result.validation.passed)
print("errors:", list(result.validation.errors)[:6])
