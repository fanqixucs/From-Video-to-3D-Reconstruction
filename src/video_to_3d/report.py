from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path


def write_scene_html(glb_path: Path, output_path: Path, title: str) -> Path:
    rel_glb = html.escape(glb_path.name)
    safe_title = html.escape(title)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; background: #111; color: #eee; font-family: system-ui, sans-serif; }}
    header {{ position: fixed; left: 20px; top: 16px; z-index: 2; }}
    h1 {{ font-size: 18px; margin: 0 0 4px; font-weight: 650; }}
    p {{ margin: 0; color: #bbb; font-size: 13px; }}
    model-viewer {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <p>VGGT-Omega point cloud with predicted camera poses</p>
  </header>
  <model-viewer src="{rel_glb}" camera-controls interaction-prompt="none" tone-mapping="neutral" shadow-intensity="0"></model-viewer>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def write_run_report(
    output_path: Path,
    run_name: str,
    artifact_paths: dict[str, Path],
    manifest: dict,
    metrics: dict,
) -> Path:
    rel = {key: path.name for key, path in artifact_paths.items()}
    generated_at = datetime.now().isoformat(timespec="seconds")
    output_path.write_text(
        "\n".join(
            [
                f"# {run_name}",
                "",
                f"Generated: `{generated_at}`",
                "",
                "## Artifacts",
                "",
                f"- Point cloud PLY: `{rel['ply']}`",
                f"- Scene GLB: `{rel['glb']}`",
                f"- Browser viewer: `{rel['html']}`",
                f"- Raw predictions: `{rel['predictions']}`",
                f"- Camera trajectory: `{rel['cameras']}`",
                f"- Input contact sheet: `{rel['contact_sheet']}`",
                f"- Depth preview: `{rel['depth_preview']}`",
                "",
                "## Reconstruction Metrics",
                "",
                "```json",
                json.dumps(metrics, indent=2),
                "```",
                "",
                "## Run Manifest",
                "",
                "```json",
                json.dumps(manifest, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path
