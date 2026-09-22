"""Minimal command-line entry points for the autonomous art pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from .asset_groups import CATEGORIES, build_catalogue
from .contracts import AssetForm, AssetRequest
from .llm import ModelPlanner, OpenAICompatibleClient
from .quality import (
    author_box_layout,
    resolve_reference_index_path,
    run_family_loop,
    run_quality_loop,
)
from .reference_index import build_index


def _parse_size(value: str | tuple[int, int] | None) -> tuple[int, int]:
    """Parse a WxH canvas request; the default stays the vanilla 16x16 icon.

    argparse applies this converter to its string default, so a handler sees an
    already-parsed tuple. Accepting both keeps the function idempotent instead
    of failing on its own output, which is exactly how a single-asset run once
    died with "expected string or bytes-like object, got 'tuple'".
    """
    if isinstance(value, tuple):
        return value
    if not value:
        return (16, 16)
    match = re.fullmatch(r"\s*(\d{1,3})\s*[xX*]\s*(\d{1,3})\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("size must look like 64x64")
    width, height = int(match.group(1)), int(match.group(2))
    if not (1 <= width <= 512 and 1 <= height <= 512):
        raise argparse.ArgumentTypeError("size must be within 1..512 on each axis")
    return (width, height)


def _query_asset_name(query: str) -> str:
    """Make a stable filesystem-safe name when the caller only gives a query."""
    ascii_slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    if ascii_slug:
        return ascii_slug[:48].rstrip("-")
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    return "asset-%s" % digest


def _plan_family(args: argparse.Namespace) -> dict | None:
    """Ask the model whether this request is one texture or a set of files.

    Whether a request needs a family ("a bow" is four files, "a bow icon" is
    one) is a content question, so it is decided by the model rather than by a
    flag the caller has to remember.
    """
    try:
        client = OpenAICompatibleClient.from_env(trace_dir=Path("outputs") / "_family_plan" / "llm_raw")
        return ModelPlanner(client).plan_family(args.query, max_members=args.max_members)
    except (RuntimeError, OSError, ValueError):
        return None


def _generate_family(
    args: argparse.Namespace,
    sources: list[str],
    index_path,
    family_plan: dict | None = None,
) -> int:
    """Generate a set of sibling textures that must share one visual language."""
    output = Path(args.out or (Path("outputs") / _query_asset_name(args.query)))
    members = None
    if args.family_members:
        members = [item.strip() for item in args.family_members.split(",") if item.strip()]
        if not members:
            print("ERROR: --family-members was empty.", file=sys.stderr)
            return 1
    try:
        family = run_family_loop(
            args.query,
            output,
            members=members,
            max_members=args.max_members,
            form=AssetForm(args.form),
            shape_policy="free" if getattr(args, "free_silhouette", False) else "planned",
            family_plan=family_plan,
            rounds=args.rounds,
            max_repairs=args.max_repairs,
            package=True,
            asset_sources=sources or None,
            recall_limit=args.recall_limit,
            reference_index_path=str(index_path) if index_path else None,
        )
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    print("QUERY -> %s" % family["query"])
    print("SET -> %s (planned by %s, %d member(s))" % (
        family["set_name"], family["plan_source"], len(family["members"])))
    if family.get("shared_style"):
        print("SHARED STYLE -> %s" % family["shared_style"])
    print("ANCHOR -> %s" % family.get("anchor"))
    for row in family["members"]:
        print("  %-24s validation=%-4s blind=%-12s palette_overlap=%s" % (
            row["name"],
            "PASS" if row["validation_passed"] else "FAIL",
            row["blind_object"] or "?",
            row.get("palette_overlap"),
        ))
        if row.get("error"):
            print("      error: %s" % row["error"])
    ok = sum(1 for row in family["members"] if row["validation_passed"])
    print("MEMBERS -> %d/%d validated" % (ok, len(family["members"])))
    continuity = family.get("continuity")
    if continuity and continuity.get("pair_count"):
        # Frame-to-frame agreement is the only thing that shows whether a set
        # of draw states reads as one object; palette overlap says nothing
        # about whether the same bow was drawn further.
        baseline = continuity.get("baseline")
        print("CONTINUITY -> exact min=%s mean=%s | near min=%s mean=%s over %d overlapping pair(s)" % (
            continuity["minimum_agreement"],
            continuity["mean_agreement"],
            continuity["minimum_near_agreement"],
            continuity["mean_near_agreement"],
            continuity["pair_count"],
        ))
        if baseline is not None:
            print("             source family exact min=%s mean=%s | near min=%s mean=%s" % (
                baseline["minimum_agreement"],
                baseline["mean_agreement"],
                baseline["minimum_near_agreement"],
                baseline["mean_near_agreement"],
            ))
    print("FAMILY -> %s" % output.resolve())
    return 0 if family["members"] and ok == len(family["members"]) else 1


def _generate(args: argparse.Namespace) -> int:
    sources = list(args.source or [])
    index_path = resolve_reference_index_path(args.reference_index)
    if not sources and not index_path:
        print(
            "ERROR: no reference evidence; pass --source <jar or resources directory>, "
            "or run index-vanilla first.",
            file=sys.stderr,
        )
        return 1

    if args.family or args.family_members:
        return _generate_family(args, sources, index_path)

    # A forced form means the caller already fixed the contract, so there is
    # nothing for the model to decide. Otherwise let it split the request: a
    # bow is four files (standby plus three draw stages), a gem is one.
    if args.form == AssetForm.AUTO.value:
        planned = _plan_family(args)
        if planned is not None and len(planned["members"]) > 1:
            print("SET -> %s (%d file(s), planned by the model)" % (
                planned["set_name"], len(planned["members"])))
            return _generate_family(args, sources, index_path, family_plan=planned)

    width, height = _parse_size(args.size)
    form = AssetForm(args.form)
    uv_layout_path = None
    if args.box_model:
        # A model that has no vanilla equivalent declares its own boxes, and the
        # atlas is derived from them instead of borrowing an unrelated layout.
        if form == AssetForm.AUTO:
            form = AssetForm.ENTITY_UV
        try:
            uv_layout_path = author_box_layout(args.query, output, max_boxes=args.max_boxes)
        except RuntimeError as exc:
            print("ERROR: %s" % exc, file=sys.stderr)
            return 1
        print("BOX MODEL -> %s" % uv_layout_path)
    request = AssetRequest(
        query=args.query,
        form=form,
        name=args.name or _query_asset_name(args.query),
        namespace=args.namespace,
        width=width,
        height=height,
        novelty=0.6,
        shape_policy="free" if getattr(args, "free_silhouette", False) else "planned",
        seed=0,
        pack_format=15,
    )
    output = Path(args.out or (Path("outputs") / request.name))
    try:
        report = run_quality_loop(
            request,
            [],
            output,
            rounds=args.rounds,
            max_repairs=args.max_repairs,
            package=True,
            reference_index_path=str(index_path) if index_path else None,
            asset_sources=sources or None,
            recall_limit=args.recall_limit,
            uv_layout_path=uv_layout_path,
        )
    except RuntimeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    selected_index = report.get("selected_round")
    selected = next(
        (round_entry for round_entry in report.get("rounds", []) if round_entry.get("index") == selected_index),
        report["rounds"][-1] if report.get("rounds") else {},
    )
    validation_passed = bool(selected.get("validation", {}).get("passed"))
    blind_aligned = bool(selected.get("blind_alignment_hint"))
    print("QUERY -> %s" % request.query)
    print("QUALITY -> %s (%d round(s))" % (output.resolve(), report.get("rounds_completed", 0)))
    print("SELECTED ROUND -> %s" % (selected.get("index", "none")))
    print("VALIDATION -> %s" % ("PASS" if validation_passed else "FAIL"))
    print("BLIND -> %s" % selected.get("blind_review", {}).get("primary_object", "unknown"))
    print("QUALITY GATE -> %s" % ("PASS" if validation_passed and blind_aligned else "FAIL"))
    return 0 if validation_passed and blind_aligned else 1


def _index_vanilla(args: argparse.Namespace) -> int:
    output = Path(args.out or (Path(__file__).resolve().parents[1] / "references" / "index.json")).expanduser().resolve()
    cache_root = Path(args.cache or (output.parent / ".cache")).expanduser().resolve()
    index = build_index(args.source, cache_root, rebuild=args.rebuild, with_pixel_text=args.with_pixel_text)
    index.save(output)
    print("INDEX -> %s" % output)
    print("SOURCE -> %s (%s)" % (index.source.path, index.source.fingerprint))
    print("TEXTURES -> %d; REUSED -> %d; BUILT -> %d; ERRORS -> %d" % (
        len(index.entries),
        index.cache_stats.get("entries_reused", 0),
        index.cache_stats.get("entries_built", 0),
        len(index.errors),
    ))
    return 0


def _list_groups(args: argparse.Namespace) -> int:
    """Show the logical asset names resolvable from the given roots.

    This is the inspection entry point for the image-free catalogue: it never
    decodes a texture, so it is a fast way to check that a block really is one
    name instead of its individual faces.
    """
    catalogue = build_catalogue(args.source)
    try:
        if args.extract:
            written = catalogue.extract(args.extract, args.to or "outputs/_groups")
            print("EXTRACT -> %s (%d texture(s))" % (args.extract, len(written)))
            for path in written:
                print("  %s" % path)
            return 0
        if args.json:
            print(json.dumps(catalogue.to_manifest(), ensure_ascii=False, indent=2))
            return 0
        stats = catalogue.stats
        print("ROOTS -> %s" % "; ".join(stats["roots"]))
        print("RESOURCE PATHS -> %d; MODELS RESOLVED -> %d" % (
            stats["resource_paths"], stats["models_resolved"]))
        groups = catalogue.select(
            category=args.category,
            namespace=args.namespace,
            contains=args.filter,
            minimum_textures=args.min_textures,
        )
        for group in groups[: args.limit]:
            print("  %s" % group.describe())
        if len(groups) > args.limit:
            print("  ... %d more (raise --limit or narrow --filter)" % (len(groups) - args.limit))
        print("SHOWN -> %d of %d selected (%d total)" % (
            min(len(groups), args.limit), len(groups), len(catalogue.groups)))
        print("BY CATEGORY -> %s" % ", ".join(
            "%s=%d" % (key, stats["groups"][key]) for key in CATEGORIES if stats["groups"].get(key)))
        print("GROUPED TEXTURES -> %s" % ", ".join(
            "%s=%d" % (key, stats["grouped_textures"][key])
            for key in CATEGORIES if stats["grouped_textures"].get(key)))
        print("NUMERIC FAMILIES -> %d (pulled in %d texture(s); folded %d frame-model group(s) into their base name)" % (
            stats["numeric_families"], stats["family_textures_pulled_in"],
            stats["family_member_groups_folded"]))
        print("ORPHANS -> %d group(s) / %d texture(s) no model references" % (
            stats["orphan_groups"], stats["orphan_textures"]))
        print("ANIMATED TEXTURES -> %d" % stats["animated_textures"])
        return 0
    finally:
        catalogue.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous Minecraft art generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate, validate and package an asset from a natural-language target")
    generate.add_argument("--query", required=True, help="the complete natural-language target")
    generate.add_argument("--out", help="output directory; defaults to outputs/<generated-name>")
    generate.add_argument("--name", help="resource name; generated from the query when omitted")
    generate.add_argument("--namespace", default="demo", help="resource-pack namespace")
    generate.add_argument("--rounds", type=int, default=3, help="quality-loop rounds")
    generate.add_argument("--max-repairs", type=int, default=1, help="repair attempts per round")
    generate.add_argument("--form", choices=[item.value for item in AssetForm], default=AssetForm.AUTO.value, help="force the render contract instead of letting the router choose; 'auto' is the default")
    generate.add_argument("--size", type=_parse_size, default="16x16", metavar="WxH", help="canvas size for a forced free-form contract, e.g. 64x64; default 16x16")
    generate.add_argument("--free-silhouette", action="store_true", help="let the model's own paint plan decide the silhouette instead of conforming to the reference model; use this when the design is meant to differ from vanilla")
    generate.add_argument("--box-model", action="store_true", help="let the model declare the object's own box decomposition and derive the UV atlas from it; implies --form entity_uv unless a form is given")
    generate.add_argument("--max-boxes", type=int, default=16, help="maximum boxes for --box-model (default 16)")
    generate.add_argument("--reference-index", help="override the default local vanilla index")
    generate.add_argument("--source", action="append", metavar="PATH", help="asset root to resolve logical reference names from live (vanilla JAR, mod JAR, resource pack, or a mod's src/main/resources); repeat for a resource-pack style stack, later wins")
    generate.add_argument("--recall-limit", type=int, help="cap how many logical names reach the router prompt; omitted sends the complete catalogue")
    generate.add_argument("--family", action="store_true", help="generate a whole set of sibling textures (armour set, tool set) that share one visual language; the member list is derived from the query")
    generate.add_argument("--family-members", help="comma-separated member suffixes for --family, e.g. 'helmet,chestplate,leggings,boots'; skips the model-planned split")
    generate.add_argument("--max-members", type=int, default=6, help="maximum members derived for --family (default 6)")
    generate.set_defaults(handler=_generate)

    index_vanilla = subparsers.add_parser("index-vanilla", help="build the local vanilla texture index used by generation")
    index_vanilla.add_argument("--source", required=True, help="Minecraft version directory, JAR, resource pack, or assets directory")
    index_vanilla.add_argument("--out", help="index JSON output; defaults to references/index.json")
    index_vanilla.add_argument("--cache", help="image cache directory; defaults to references/.cache")
    index_vanilla.add_argument("--rebuild", action="store_true", help="rebuild cached texture data")
    index_vanilla.add_argument("--with-pixel-text", action="store_true", help="store pixel-text summaries for every texture")
    index_vanilla.set_defaults(handler=_index_vanilla)

    list_groups = subparsers.add_parser("list-groups", help="list logical asset names (one name = all of its textures) resolved live from one or more asset roots")
    list_groups.add_argument("--source", action="append", required=True, metavar="PATH", help="vanilla JAR, mod JAR, resource pack, or directory with assets/; repeat for a resource-pack style stack (later wins)")
    list_groups.add_argument("--category", choices=list(CATEGORIES), help="restrict to block/item/entity/texture")
    list_groups.add_argument("--namespace", help="restrict to one asset namespace, e.g. minecraft or a mod id")
    list_groups.add_argument("--filter", help="case-insensitive substring match on the asset name")
    list_groups.add_argument("--min-textures", type=int, default=0, help="only names owning at least this many textures")
    list_groups.add_argument("--limit", type=int, default=40, help="maximum rows to print")
    list_groups.add_argument("--json", action="store_true", help="print the full router-facing manifest as JSON")
    list_groups.add_argument("--extract", metavar="ASSET_ID", help="write every texture of one name to --to and exit")
    list_groups.add_argument("--to", help="destination directory for --extract")
    list_groups.set_defaults(handler=_list_groups)
    render = subparsers.add_parser("render", help="execute a plan.json with no model calls: rasterise, validate and package it; the caller owns every model decision")
    render.add_argument("--plan", required=True, help="a plan.json, either written by an earlier generate run or authored by the caller")
    render.add_argument("--out", required=True, help="output directory")
    render.add_argument("--no-package", action="store_true", help="skip resource-pack assembly")
    render.set_defaults(handler=_render_plan)

    evidence = subparsers.add_parser("evidence", help="print everything needed to author a plan for one logical asset: names, roles, sizes and literal pixels of every reference")
    evidence.add_argument("--source", action="append", required=True, metavar="PATH", help="vanilla JAR, mod JAR, resource pack, or directory with assets/; repeat for a resource-pack style stack (later wins)")
    evidence.add_argument("--name", required=True, help="logical asset name, e.g. bow, clock, oak_log")
    evidence.add_argument("--member", help="the family member this request is about, e.g. bow_standby; its siblings are demoted to context")
    evidence.add_argument("--cache", help="cache directory; defaults to references/.cache/live")
    evidence.add_argument("--max-frames", type=int, default=8, help="cap how many frames of one code-driven family are expanded (default 8)")
    evidence.add_argument("--json", action="store_true", help="print a machine-readable summary instead of the prompt text")
    evidence.set_defaults(handler=_print_evidence)
    return parser



class _OfflineCritic:
    """No model: a plan handed to 'render' is already judged by its author."""

    repair_soft_failures = False

    def review(self, descriptor, compiled):
        from .validation import ValidationResult

        return ValidationResult(
            passed=True,
            stage="offline_render",
            metrics={"semantic_review_skipped": True},
            warnings=["render executes a plan; no semantic model was consulted"],
        )


def _render_plan(args: argparse.Namespace) -> int:
    """Execute a plan file: the deterministic half of the pipeline, alone.

    Every model decision in this project is already frozen into one plan.json
    (request, descriptor, geometry, appearance, references). Rendering it needs
    no API key, no network and no judgement, which is what lets an agent own
    the model half and this command own the rest.
    """
    from .pipeline import GenerationPipeline
    from .plans import plan_from_file

    plan = plan_from_file(args.plan)
    result = GenerationPipeline(
        critic=_OfflineCritic(),
        repairer=None,
        max_geometry_repairs=0,
    ).run(plan, args.out, package=not args.no_package)
    print("RENDER -> %s" % Path(args.out).resolve())
    print("REQUEST -> %s (%dx%d)" % (plan.request.name or plan.request.query, plan.request.width, plan.request.height))
    print("PARTS -> %s" % ", ".join(part.id for part in plan.descriptor.parts))
    print("SPRITE -> %s" % result.sprite_path)
    print("VALIDATION -> %s" % ("passed" if result.validation.passed else "FAILED"))
    for error in list(result.validation.errors)[:5]:
        print("  error: %s" % error)
    return 0 if result.sprite_path and result.validation.passed else 1


def _print_evidence(args: argparse.Namespace) -> int:
    """Print the reference evidence a plan author needs, with no model call.

    This is the other half of 'render': it hands back exactly what the planner
    prompts used to assemble -- which references exist, what role each carries,
    which one answers this request, and the literal pixels of every small
    raster -- so a caller can write the descriptor, geometry and appearance
    itself.
    """
    import json as _json

    from .asset_groups import build_catalogue
    from .group_index import GroupReferenceSource
    from .llm import _appearance_reference_evidence
    from .contracts import ReferenceAsset, ReferenceRole

    catalogue = build_catalogue(args.source)
    group = next((item for item in catalogue.select() if item.name == args.name), None)
    if group is None:
        names = sorted({item.name for item in catalogue.select()})
        close = [name for name in names if args.name.lower() in name.lower()][:10]
        print("ERROR: no logical asset named %r" % args.name, file=sys.stderr)
        if close:
            print("       did you mean: %s" % ", ".join(close), file=sys.stderr)
        return 1
    source = GroupReferenceSource(catalogue, args.cache or (Path("references") / ".cache" / "live"))
    entry = source.entry_for(group.asset_id)
    assets = source.planning_assets(
        entry,
        [ReferenceRole.SHAPE, ReferenceRole.PIXEL_STYLE],
        display_name=args.name,
        preferred_member=args.member,
        max_frames=args.max_frames,
    )
    # Show the same roles a generation run would attach: one frame answers this
    # request, its siblings are context. Printing all of them as 'shape' is what
    # let a standby bow copy the half-drawn frame's silhouette.
    from .quality import _shape_authority

    authority = _shape_authority(assets, args.member)
    if authority is not None:
        marked = []
        for asset in assets:
            if asset is authority:
                roles = list(asset.roles)
                notes = list(asset.notes) + ["shape_authority=this request; copy this silhouette"]
            else:
                roles = [r for r in asset.roles if r is not ReferenceRole.SHAPE] or [ReferenceRole.PIXEL_STYLE]
                notes = list(asset.notes) + [
                    "shape_authority=context only; another state of the same object, "
                    "not this request's silhouette"
                ]
            marked.append(ReferenceAsset(path=asset.path, name=asset.name, roles=roles, notes=notes, features=asset.features))
        assets = marked
    if args.json:
        print(_json.dumps({
            "asset_id": group.asset_id,
            "member": args.member,
            "reference_count": len(assets),
            "references": [
                {
                    "name": asset.name,
                    "path": asset.path,
                    "roles": [role.value for role in asset.roles],
                    "notes": list(asset.notes),
                    "size": [asset.features.get("width"), asset.features.get("height")],
                }
                for asset in assets
            ],
        }, ensure_ascii=False, indent=2))
        return 0
    print("ASSET -> %s (%d reference(s))" % (group.asset_id, len(assets)))
    print()
    print(_appearance_reference_evidence(assets))
    return 0

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
