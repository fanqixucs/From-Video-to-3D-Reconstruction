from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SemanticClass:
    label_id: int
    name: str
    prompts: tuple[str, ...]
    color: tuple[int, int, int]
    priority: int = 1


DEFAULT_CLASSES = (
    SemanticClass(1, "wall", ("wall",), (183, 189, 196), 0),
    SemanticClass(2, "bulletin board", ("bulletin board", "notice board", "cork board"), (220, 121, 67), 2),
    SemanticClass(3, "cupboard", ("cupboard", "cabinet", "kitchen cabinet"), (231, 204, 112), 2),
    SemanticClass(4, "kettle", ("kettle", "electric kettle"), (80, 88, 98), 2),
    SemanticClass(5, "tea box", ("tea box", "tea boxes", "yellow box", "box"), (241, 214, 64), 2),
    SemanticClass(6, "sanitizer", ("sanitizer", "sanitizer bottle", "soap dispenser"), (113, 170, 217), 2),
    SemanticClass(7, "tissue box", ("tissue box", "tissue", "paper towel"), (169, 220, 210), 2),
    SemanticClass(8, "tap", ("tap", "faucet"), (112, 189, 134), 2),
    SemanticClass(9, "cleaning solution", ("cleaning solution", "dish soap", "soap bottle", "detergent bottle"), (105, 190, 110), 2),
    SemanticClass(10, "coffee machine", ("coffee machine", "coffee maker", "coffee pod machine"), (62, 70, 82), 2),
    SemanticClass(11, "microwave", ("microwave", "microwave oven"), (230, 230, 225), 2),
    SemanticClass(12, "heater", ("heater", "radiator"), (210, 210, 205), 1),
    SemanticClass(13, "dust bin", ("dust bin", "trash bin", "rubbish bin", "bin"), (137, 214, 70), 2),
    SemanticClass(14, "sink", ("sink", "basin"), (178, 196, 204), 1),
    SemanticClass(15, "countertop", ("countertop", "counter top", "worktop", "kitchen counter"), (86, 86, 82), 1),
    SemanticClass(16, "floor", ("floor",), (124, 137, 181), 0),
    SemanticClass(17, "ceiling", ("ceiling",), (232, 235, 238), 0),
    SemanticClass(18, "door", ("door",), (196, 178, 135), 1),
)


def class_maps(classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES):
    id_to_class = {cls.label_id: cls for cls in classes}
    prompt_to_class = {}
    for cls in classes:
        for prompt in cls.prompts:
            prompt_to_class[prompt.lower()] = cls
    return id_to_class, prompt_to_class


def images_from_predictions(predictions: dict[str, np.ndarray]) -> np.ndarray:
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    return (images * 255).clip(0, 255).astype(np.uint8)


class GroundedSAMSemanticSegmenter:
    def __init__(
        self,
        classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_id: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        use_sam: bool = True,
    ) -> None:
        self.classes = classes
        self.detector_id = detector_id
        self.sam_id = sam_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_sam = use_sam
        self.id_to_class, self.prompt_to_class = class_maps(classes)
        self._detector_processor = None
        self._detector_model = None
        self._sam_processor = None
        self._sam_model = None
        self._torch = None

    def load(self) -> None:
        if self._detector_model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self._detector_processor = AutoProcessor.from_pretrained(self.detector_id)
        self._detector_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_id).to(self.device).eval()
        )
        if self.use_sam:
            from transformers import SamModel, SamProcessor

            self._sam_processor = SamProcessor.from_pretrained(self.sam_id)
            self._sam_model = SamModel.from_pretrained(self.sam_id).to(self.device).eval()

    @property
    def prompt(self) -> str:
        prompts = []
        for cls in self.classes:
            prompts.extend(cls.prompts)
        return ". ".join(prompts) + "."

    def label_frame(self, image_rgb: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        self.load()
        detections = self._detect(image_rgb)
        label_map = -np.ones(image_rgb.shape[:2], dtype=np.int32)
        if not detections:
            return label_map, []

        boxes = np.stack([det["box"] for det in detections], axis=0)
        masks = self._segment(image_rgb, boxes) if self.use_sam else self._box_masks(image_rgb.shape[:2], boxes)

        ordered = sorted(
            zip(detections, masks),
            key=lambda item: (item[0]["priority"], item[0]["score"]),
        )
        for det, mask in ordered:
            label_map[mask] = det["label_id"]
            det["area_px"] = int(mask.sum())
        return label_map, detections

    def _detect(self, image_rgb: np.ndarray) -> list[dict]:
        pil_image = Image.fromarray(image_rgb)
        inputs = self._detector_processor(images=pil_image, text=self.prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self._detector_model(**inputs)
        h, w = image_rgb.shape[:2]
        try:
            results = self._detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]
        except TypeError:
            results = self._detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]

        label_texts = results.get("text_labels", results.get("labels", []))
        detections = []
        for box, score, label_text in zip(results["boxes"], results["scores"], label_texts):
            cls = self._canonical_class(str(label_text).lower())
            if cls is None:
                continue
            box_np = np.asarray(box.detach().cpu().numpy(), dtype=np.float32)
            detections.append(
                {
                    "label_id": cls.label_id,
                    "label": cls.name,
                    "score": float(score.detach().cpu().numpy()),
                    "box_xyxy": box_np.tolist(),
                    "box": box_np,
                    "text": str(label_text),
                    "priority": cls.priority,
                }
            )
        return detections

    def _canonical_class(self, text: str) -> SemanticClass | None:
        for prompt, cls in self.prompt_to_class.items():
            if prompt in text or text in prompt:
                return cls
        return None

    def _segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        if len(boxes_xyxy) == 0:
            return np.zeros((0, *image_rgb.shape[:2]), dtype=bool)
        pil_image = Image.fromarray(image_rgb)
        inputs = self._sam_processor(pil_image, input_boxes=[boxes_xyxy.tolist()], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self._sam_model(**inputs, multimask_output=False)
        masks = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        return masks.squeeze(1).numpy().astype(bool)

    @staticmethod
    def _box_masks(image_shape: tuple[int, int], boxes_xyxy: np.ndarray) -> np.ndarray:
        h, w = image_shape
        masks = np.zeros((len(boxes_xyxy), h, w), dtype=bool)
        for idx, box in enumerate(boxes_xyxy.astype(int)):
            x1, y1, x2, y2 = box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 > x1 and y2 > y1:
                masks[idx, y1:y2, x1:x2] = True
        return masks


def lift_semantics_to_points(
    predictions: dict[str, np.ndarray],
    label_maps: np.ndarray,
    frame_indices: list[int],
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
    min_conf_percentile: float = 10.0,
    max_points_per_label: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    points = predictions["world_points_from_depth"]
    images = images_from_predictions(predictions)
    conf = predictions["depth_conf"]
    id_to_class, _ = class_maps(classes)
    rng = np.random.default_rng(7)

    all_points = []
    all_rgb = []
    all_labels = []
    all_source_frame = []
    all_source_y = []
    all_source_x = []
    all_source_conf = []
    summary = {}

    for label_id, cls in id_to_class.items():
        label_points = []
        label_rgb = []
        label_source_frame = []
        label_source_y = []
        label_source_x = []
        label_source_conf = []
        for map_idx, frame_idx in enumerate(frame_indices):
            label_mask = label_maps[map_idx] == label_id
            if not label_mask.any():
                continue
            frame_points = points[frame_idx]
            frame_conf = conf[frame_idx]
            finite = np.isfinite(frame_points).all(axis=-1) & np.isfinite(frame_conf)
            threshold = np.percentile(frame_conf[finite], min_conf_percentile) if finite.any() else 0.0
            mask = label_mask & finite & (frame_conf >= threshold)
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            label_points.append(frame_points[mask])
            label_rgb.append(images[frame_idx][mask])
            label_source_frame.append(np.full(len(ys), frame_idx, dtype=np.int32))
            label_source_y.append(ys.astype(np.int32))
            label_source_x.append(xs.astype(np.int32))
            label_source_conf.append(frame_conf[mask].astype(np.float32))

        if not label_points:
            summary[str(label_id)] = {"name": cls.name, "points": 0, "color": cls.color}
            continue

        lp = np.concatenate(label_points, axis=0)
        lr = np.concatenate(label_rgb, axis=0)
        lsf = np.concatenate(label_source_frame, axis=0)
        lsy = np.concatenate(label_source_y, axis=0)
        lsx = np.concatenate(label_source_x, axis=0)
        lsc = np.concatenate(label_source_conf, axis=0)
        if max_points_per_label > 0 and len(lp) > max_points_per_label:
            keep = rng.choice(len(lp), max_points_per_label, replace=False)
            lp = lp[keep]
            lr = lr[keep]
            lsf = lsf[keep]
            lsy = lsy[keep]
            lsx = lsx[keep]
            lsc = lsc[keep]

        all_points.append(lp.astype(np.float32))
        all_rgb.append(lr.astype(np.uint8))
        all_labels.append(np.full(len(lp), label_id, dtype=np.int32))
        all_source_frame.append(lsf)
        all_source_y.append(lsy)
        all_source_x.append(lsx)
        all_source_conf.append(lsc)
        summary[str(label_id)] = {
            "name": cls.name,
            "points": int(len(lp)),
            "color": cls.color,
            "source_frames": int(len(np.unique(lsf))),
            "source_conf_median": float(np.median(lsc)),
        }

    if not all_points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int32),
            {
                "frame": np.zeros((0,), dtype=np.int32),
                "y": np.zeros((0,), dtype=np.int32),
                "x": np.zeros((0,), dtype=np.int32),
                "depth_conf": np.zeros((0,), dtype=np.float32),
            },
            summary,
        )
    sources = {
        "frame": np.concatenate(all_source_frame),
        "y": np.concatenate(all_source_y),
        "x": np.concatenate(all_source_x),
        "depth_conf": np.concatenate(all_source_conf),
    }
    return np.concatenate(all_points), np.concatenate(all_rgb), np.concatenate(all_labels), sources, summary


def write_semantic_ply(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
) -> Path:
    id_to_class, _ = class_maps(classes)
    colors = np.array([id_to_class[int(label)].color for label in labels], dtype=np.uint8)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("semantic_id", "<i4"),
        ]
    )
    data = np.empty(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]
    data["semantic_id"] = labels

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment semantic_id maps are listed below; most PLY viewers display colors but not text annotations",
            *[f"comment semantic_id {cls.label_id}: {cls.name}" for cls in classes],
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int semantic_id",
            "end_header",
            "",
        ]
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(data.tobytes())
    return path


def write_semantic_ply_with_text_labels(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    summary: dict,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
) -> Path:
    id_to_class, _ = class_maps(classes)
    base_colors = np.array([id_to_class[int(label)].color for label in labels], dtype=np.uint8)
    text_points, text_colors, text_labels = build_text_label_points(points, centroids, summary, classes)

    if len(text_points):
        all_points = np.concatenate([points.astype(np.float32, copy=False), text_points.astype(np.float32, copy=False)], axis=0)
        all_colors = np.concatenate([base_colors, text_colors], axis=0)
        all_labels = np.concatenate([labels.astype(np.int32, copy=False), text_labels.astype(np.int32, copy=False)], axis=0)
        is_label = np.concatenate([np.zeros(len(points), dtype=np.uint8), np.ones(len(text_points), dtype=np.uint8)])
    else:
        all_points = points.astype(np.float32, copy=False)
        all_colors = base_colors
        all_labels = labels.astype(np.int32, copy=False)
        is_label = np.zeros(len(points), dtype=np.uint8)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("semantic_id", "<i4"),
            ("is_text_label", "u1"),
        ]
    )
    data = np.empty(len(all_points), dtype=dtype)
    data["x"] = all_points[:, 0]
    data["y"] = all_points[:, 1]
    data["z"] = all_points[:, 2]
    data["red"] = all_colors[:, 0]
    data["green"] = all_colors[:, 1]
    data["blue"] = all_colors[:, 2]
    data["semantic_id"] = all_labels
    data["is_text_label"] = is_label

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment semantic labels are baked into this PLY as additional 3D point-stroke text",
            "comment semantic_id maps are listed below",
            *[f"comment semantic_id {cls.label_id}: {cls.name}" for cls in classes],
            f"element vertex {len(all_points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int semantic_id",
            "property uchar is_text_label",
            "end_header",
            "",
        ]
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(data.tobytes())
    return path


def write_semantic_ply_with_corner_legend(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    summary: dict,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
) -> Path:
    id_to_class, _ = class_maps(classes)
    base_colors = np.array([id_to_class[int(label)].color for label in labels], dtype=np.uint8)
    legend_points, legend_colors, legend_labels = build_corner_legend_points(points, summary, classes)

    if len(legend_points):
        all_points = np.concatenate([points.astype(np.float32, copy=False), legend_points], axis=0)
        all_colors = np.concatenate([base_colors, legend_colors], axis=0)
        all_labels = np.concatenate([labels.astype(np.int32, copy=False), legend_labels], axis=0)
        is_legend = np.concatenate([np.zeros(len(points), dtype=np.uint8), np.ones(len(legend_points), dtype=np.uint8)])
    else:
        all_points = points.astype(np.float32, copy=False)
        all_colors = base_colors
        all_labels = labels.astype(np.int32, copy=False)
        is_legend = np.zeros(len(points), dtype=np.uint8)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("semantic_id", "<i4"),
            ("is_corner_legend", "u1"),
        ]
    )
    data = np.empty(len(all_points), dtype=dtype)
    data["x"] = all_points[:, 0]
    data["y"] = all_points[:, 1]
    data["z"] = all_points[:, 2]
    data["red"] = all_colors[:, 0]
    data["green"] = all_colors[:, 1]
    data["blue"] = all_colors[:, 2]
    data["semantic_id"] = all_labels
    data["is_corner_legend"] = is_legend

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment semantic class legend is baked into one scene corner as 3D point-stroke text",
            "comment semantic_id maps are listed below",
            *[f"comment semantic_id {cls.label_id}: {cls.name}" for cls in classes],
            f"element vertex {len(all_points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int semantic_id",
            "property uchar is_corner_legend",
            "end_header",
            "",
        ]
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(data.tobytes())
    return path


def build_text_label_points(
    scene_points: np.ndarray,
    centroids: dict[int, np.ndarray],
    summary: dict,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    id_to_class, _ = class_maps(classes)
    finite = np.isfinite(scene_points).all(axis=1)
    if finite.any():
        lo, hi = np.percentile(scene_points[finite], [2, 98], axis=0)
        scene_diag = float(np.linalg.norm(hi - lo))
    else:
        scene_diag = 1.0
    scene_diag = max(scene_diag, 1e-3)

    right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    toward_viewer = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    text_height = 0.030 * scene_diag
    label_gap = 0.045 * scene_diag

    all_points = []
    all_colors = []
    all_labels = []
    visible_ids = [
        label_id
        for label_id in sorted(centroids)
        if summary.get(str(label_id), {}).get("points", 0) > 0 and label_id in id_to_class
    ]

    for rank, label_id in enumerate(visible_ids):
        cls = id_to_class[label_id]
        centroid = centroids[label_id].astype(np.float32)
        stagger = ((rank % 5) - 2) * 0.18 * text_height
        origin = centroid + toward_viewer * label_gap + up * (1.6 * text_height + stagger)
        glyph = text_to_point_billboard(cls.name, origin, cls.color, text_height, right=right, up=up)
        leader = leader_line_points(centroid, origin - up * (0.45 * text_height), cls.color, label_id, steps=72)
        if len(glyph[0]):
            all_points.append(glyph[0])
            all_colors.append(glyph[1])
            all_labels.append(np.full(len(glyph[0]), label_id, dtype=np.int32))
        if len(leader[0]):
            all_points.append(leader[0])
            all_colors.append(leader[1])
            all_labels.append(leader[2])

    if not all_points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.int32),
        )
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), np.concatenate(all_labels, axis=0)


def build_corner_legend_points(
    scene_points: np.ndarray,
    summary: dict,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    id_to_class, _ = class_maps(classes)
    finite = np.isfinite(scene_points).all(axis=1)
    if finite.any():
        lo, hi = np.percentile(scene_points[finite], [1, 99], axis=0)
    else:
        lo = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        hi = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    extent = np.maximum(hi - lo, 1e-3)
    scene_diag = float(np.linalg.norm(extent))

    right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    row_h = 0.026 * scene_diag
    text_h = 0.017 * scene_diag
    swatch = 0.012 * scene_diag
    origin = np.array(
        [
            lo[0] - 0.08 * scene_diag,
            lo[1] - 0.14 * scene_diag,
            hi[2] + 0.06 * scene_diag,
        ],
        dtype=np.float32,
    )

    rows = [
        (label_id, id_to_class[label_id], summary[str(label_id)])
        for label_id in sorted(id_to_class)
        if summary.get(str(label_id), {}).get("points", 0) > 0
    ]
    all_points = []
    all_colors = []
    all_labels = []

    title_points, title_colors = text_to_point_billboard(
        "semantic legend",
        origin,
        (255, 255, 255),
        text_h * 1.18,
        right=right,
        up=up,
    )
    if len(title_points):
        all_points.append(title_points)
        all_colors.append(title_colors)
        all_labels.append(np.full(len(title_points), -1, dtype=np.int32))

    for row_idx, (label_id, cls, item) in enumerate(rows):
        row_origin = origin - up * ((row_idx + 1.45) * row_h)
        swatch_points, swatch_colors = square_points(
            row_origin,
            right=right,
            up=up,
            size=swatch,
            color=cls.color,
            samples=16,
        )
        count = int(item.get("points", 0))
        name = f"{label_id:02d} {cls.name} {count // 1000}k"
        text_points, text_colors = text_to_point_billboard(
            name,
            row_origin + right * (swatch * 2.8),
            cls.color,
            text_h,
            right=right,
            up=up,
        )
        if len(swatch_points):
            all_points.append(swatch_points)
            all_colors.append(swatch_colors)
            all_labels.append(np.full(len(swatch_points), label_id, dtype=np.int32))
        if len(text_points):
            all_points.append(text_points)
            all_colors.append(text_colors)
            all_labels.append(np.full(len(text_points), label_id, dtype=np.int32))

    if not all_points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.int32),
        )
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), np.concatenate(all_labels, axis=0)


def square_points(
    center: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    size: float,
    color: tuple[int, int, int],
    samples: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-0.5, 0.5, samples, dtype=np.float32) * size
    xx, zz = np.meshgrid(coords, coords, indexing="xy")
    points = center[None] + xx.reshape(-1, 1) * right[None] + zz.reshape(-1, 1) * up[None]
    colors = np.repeat(np.array(color, dtype=np.uint8)[None], len(points), axis=0)
    return points.astype(np.float32), colors


def text_to_point_billboard(
    text: str,
    origin: np.ndarray,
    color: tuple[int, int, int],
    text_height: float,
    right: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    pad = 8
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    mask = np.zeros((text_h + baseline + 2 * pad, text_w + 2 * pad), dtype=np.uint8)
    cv2.putText(mask, text, (pad, pad + text_h), font, font_scale, 255, thickness, cv2.LINE_AA)
    ys, xs = np.nonzero(mask > 20)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    scale = text_height / max(mask.shape[0], 1)
    local_x = (xs.astype(np.float32) - mask.shape[1] * 0.5) * scale
    local_z = (mask.shape[0] * 0.5 - ys.astype(np.float32)) * scale
    points = origin[None] + local_x[:, None] * right[None] + local_z[:, None] * up[None]
    colors = np.repeat(np.array(color, dtype=np.uint8)[None], len(points), axis=0)
    return points.astype(np.float32), colors


def leader_line_points(
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    label_id: int,
    steps: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps <= 1:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.int32),
        )
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    points = (1.0 - alpha) * start.astype(np.float32)[None] + alpha * end.astype(np.float32)[None]
    line_color = np.array(color, dtype=np.float32) * 0.45 + np.array([255, 255, 255], dtype=np.float32) * 0.55
    colors = np.repeat(line_color.clip(0, 255).astype(np.uint8)[None], steps, axis=0)
    labels = np.full(steps, label_id, dtype=np.int32)
    return points.astype(np.float32), colors, labels


def align_points_to_scene(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    extrinsics = np.zeros((len(extrinsic), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = extrinsic
    extrinsics[:, 3, 3] = 1.0
    opengl = np.eye(4)
    opengl[1, 1] = -1
    opengl[2, 2] = -1
    transform = np.linalg.inv(extrinsics[0]) @ opengl
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def write_overlay_sheet(
    images: np.ndarray,
    label_maps: np.ndarray,
    frame_indices: list[int],
    path: Path,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
    max_tiles: int = 12,
) -> Path:
    id_to_class, _ = class_maps(classes)
    tiles = []
    for local_idx, frame_idx in enumerate(frame_indices[:max_tiles]):
        image = images[frame_idx].copy()
        label_map = label_maps[local_idx]
        overlay = image.copy()
        for label_id, cls in id_to_class.items():
            mask = label_map == label_id
            if mask.any():
                overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(cls.color)).astype(np.uint8)
        tile = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(tile, f"frame {frame_idx:02d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(tile, f"frame {frame_idx:02d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        tiles.append(tile)

    if not tiles:
        raise RuntimeError("No semantic overlay tiles to write")

    thumb_w = 240
    resized = []
    for tile in tiles:
        scale = thumb_w / tile.shape[1]
        resized.append(cv2.resize(tile, (thumb_w, int(round(tile.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    cols = min(4, len(resized))
    rows = int(np.ceil(len(resized) / cols))
    thumb_h = max(tile.shape[0] for tile in resized)
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(resized):
        y = (idx // cols) * thumb_h
        x = (idx % cols) * thumb_w
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return path


def semantic_centroids(points_scene: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centroids = {}
    for label_id in sorted(set(labels.astype(int).tolist())):
        mask = labels == label_id
        if not mask.any():
            continue
        centroids[label_id] = np.median(points_scene[mask], axis=0)
    return centroids


def write_semantic_legend(
    path: Path,
    summary: dict,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
) -> Path:
    id_to_class, _ = class_maps(classes)
    rows = [(label_id, id_to_class[label_id], summary[str(label_id)]) for label_id in id_to_class if summary.get(str(label_id), {}).get("points", 0) > 0]
    width = 900
    row_h = 34
    height = 72 + row_h * len(rows)
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(image, "Geometry-aligned semantic labels", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(image, "Each label is lifted from a 2D mask through VGGT world_points_from_depth.", (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    y = 92
    for label_id, cls, item in rows:
        color_bgr = tuple(int(c) for c in cls.color[::-1])
        cv2.rectangle(image, (24, y - 20), (54, y + 8), color_bgr, -1)
        cv2.rectangle(image, (24, y - 20), (54, y + 8), (40, 40, 40), 1)
        text = f"{label_id:02d}  {cls.name:<18}  {item['points']:,} pts  {item.get('source_frames', 0)} frames  conf {item.get('source_conf_median', 0):.2f}"
        cv2.putText(image, text, (68, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (30, 30, 30), 1, cv2.LINE_AA)
        y += row_h
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return path


def write_labeled_semantic_html(
    path: Path,
    semantic_ply_name: str,
    centroids: dict[int, np.ndarray],
    summary: dict,
    classes: tuple[SemanticClass, ...] = DEFAULT_CLASSES,
) -> Path:
    id_to_class, _ = class_maps(classes)
    labels = []
    for label_id, centroid in centroids.items():
        item = summary.get(str(label_id), {})
        if item.get("points", 0) <= 0:
            continue
        cls = id_to_class[label_id]
        labels.append(
            {
                "id": label_id,
                "name": cls.name,
                "position": [float(x) for x in centroid],
                "color": f"rgb({cls.color[0]},{cls.color[1]},{cls.color[2]})",
                "points": int(item.get("points", 0)),
            }
        )
    labels_json = json.dumps(labels, indent=2)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic 3D Labels</title>
  <style>
    html, body {{ margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: system-ui, sans-serif; }}
    #viewer {{ position: fixed; inset: 0; }}
    #title {{ position: fixed; top: 16px; left: 18px; z-index: 2; }}
    #title h1 {{ margin: 0 0 4px; font-size: 18px; }}
    #title p {{ margin: 0; color: #bbb; font-size: 13px; }}
    #status {{ margin-top: 6px; color: #92d8ff; font-size: 12px; }}
    #legend {{ position: fixed; right: 16px; top: 16px; width: 260px; max-height: calc(100vh - 32px); overflow: auto; background: rgba(20,20,20,0.78); border: 1px solid #555; padding: 10px; z-index: 2; }}
    .row {{ display: grid; grid-template-columns: 16px 1fr auto; gap: 8px; align-items: center; font-size: 12px; margin: 6px 0; }}
    .swatch {{ width: 14px; height: 14px; border: 1px solid #ddd; }}
    .label {{ position: absolute; padding: 3px 6px; border-radius: 3px; color: #111; font-size: 12px; font-weight: 700; pointer-events: none; white-space: nowrap; transform: translate(-50%, -50%); box-shadow: 0 1px 4px rgba(0,0,0,0.45); }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="title">
    <h1>Semantic 3D Labels</h1>
    <p>Labels are lifted through VGGT geometry; click/drag to inspect.</p>
    <div id="status">Loading semantic point cloud...</div>
  </div>
  <div id="legend"></div>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import {{ PLYLoader }} from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js';

    const labels = {labels_json};
    const container = document.getElementById('viewer');
    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(0.0, -1.2, 1.0);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, -0.7);

    scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const axes = new THREE.AxesHelper(0.25);
    scene.add(axes);

    const labelEls = [];
    const legend = document.getElementById('legend');
    legend.innerHTML = '<b>Legend</b>';
    for (const item of labels) {{
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<span class="swatch" style="background:${{item.color}}"></span><span>${{item.name}}</span><span>${{(item.points/1000).toFixed(0)}}k</span>`;
      legend.appendChild(row);

      const el = document.createElement('div');
      el.className = 'label';
      el.textContent = item.name;
      el.style.background = item.color;
      document.body.appendChild(el);
      labelEls.push({{ el, position: new THREE.Vector3(...item.position) }});
    }}

    new PLYLoader().load('{semantic_ply_name}', geometry => {{
      geometry.computeBoundingSphere();
      const material = new THREE.PointsMaterial({{ size: 2.0, sizeAttenuation: false, vertexColors: true }});
      const points = new THREE.Points(geometry, material);
      scene.add(points);
      status.textContent = `Loaded ${{geometry.attributes.position.count.toLocaleString()}} semantic points`;
      if (geometry.boundingSphere) {{
        controls.target.copy(geometry.boundingSphere.center);
        const radius = Math.max(geometry.boundingSphere.radius, 0.4);
        camera.position.copy(geometry.boundingSphere.center).add(new THREE.Vector3(0.0, -radius * 1.8, radius * 0.8));
        camera.near = Math.max(radius / 1000, 0.001);
        camera.far = Math.max(radius * 20, 10);
        camera.updateProjectionMatrix();
        controls.update();
      }}
    }}, undefined, error => {{
      status.textContent = `Point cloud failed to load: ${{error.message || error}}`;
      status.style.color = '#ff9b9b';
    }});

    function updateLabels() {{
      for (const item of labelEls) {{
        const p = item.position.clone().project(camera);
        const visible = p.z > -1 && p.z < 1;
        item.el.style.display = visible ? 'block' : 'none';
        item.el.style.left = `${{(p.x * 0.5 + 0.5) * window.innerWidth}}px`;
        item.el.style.top = `${{(-p.y * 0.5 + 0.5) * window.innerHeight}}px`;
      }}
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      updateLabels();
    }}
    animate();

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add geometry-aligned semantic labels to a VGGT reconstruction.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Reconstruction run directory.")
    parser.add_argument("--predictions", type=Path, default=None, help="Defaults to <run-dir>/predictions.npz.")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-semantic-frames", type=int, default=20)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--detector-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-id", default="facebook/sam-vit-huge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--max-points-per-label", type=int, default=250_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predictions_path = args.predictions or (args.run_dir / "predictions.npz")
    with np.load(predictions_path) as loaded:
        predictions = {key: np.array(loaded[key]) for key in loaded.files}
    images = images_from_predictions(predictions)
    frame_indices = list(range(0, len(images), args.frame_stride))[: args.max_semantic_frames]

    segmenter = GroundedSAMSemanticSegmenter(
        detector_id=args.detector_id,
        sam_id=args.sam_id,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        use_sam=not args.no_sam,
    )

    label_maps = []
    detections_by_frame = {}
    for frame_idx in frame_indices:
        print(f"[semantic] frame {frame_idx}", flush=True)
        label_map, detections = segmenter.label_frame(images[frame_idx])
        label_maps.append(label_map)
        detections_by_frame[str(frame_idx)] = [
            {key: value for key, value in det.items() if key not in {"box", "priority"}}
            for det in detections
        ]
    label_maps_arr = np.stack(label_maps, axis=0)

    points, rgb, labels, sources, summary = lift_semantics_to_points(
        predictions,
        label_maps_arr,
        frame_indices,
        max_points_per_label=args.max_points_per_label,
    )

    semantic_dir = args.run_dir / "semantics"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    points_scene = align_points_to_scene(points, predictions["extrinsic"])
    np.savez_compressed(
        semantic_dir / "semantic_labeled_points.npz",
        points_scene=points_scene,
        points_world_raw=points,
        rgb=rgb,
        label_id=labels,
        source_frame=sources["frame"],
        source_y=sources["y"],
        source_x=sources["x"],
        source_depth_conf=sources["depth_conf"],
        frame_indices=np.array(frame_indices, dtype=np.int32),
        label_maps=label_maps_arr,
    )
    ply_path = write_semantic_ply(semantic_dir / "semantic_pointcloud.ply", points_scene, labels)
    overlay_path = write_overlay_sheet(images, label_maps_arr, frame_indices, semantic_dir / "semantic_overlays.jpg")
    legend_path = write_semantic_legend(semantic_dir / "semantic_legend.jpg", summary)
    centroids = semantic_centroids(points_scene, labels)
    labeled_ply_path = write_semantic_ply_with_text_labels(
        semantic_dir / "semantic_pointcloud_with_labels.ply",
        points_scene,
        labels,
        centroids,
        summary,
    )
    corner_legend_ply_path = write_semantic_ply_with_corner_legend(
        semantic_dir / "semantic_pointcloud_with_corner_legend.ply",
        points_scene,
        labels,
        summary,
    )
    html_path = write_labeled_semantic_html(
        semantic_dir / "semantic_labeled_viewer.html",
        semantic_ply_name=ply_path.name,
        centroids=centroids,
        summary=summary,
    )

    class_payload = {
        str(cls.label_id): {"name": cls.name, "prompts": cls.prompts, "color": cls.color}
        for cls in DEFAULT_CLASSES
    }
    payload = {
        "method": "GroundingDINO text detection + SAM masks on VGGT preprocessed frames; masks lifted through world_points_from_depth.",
        "alignment_statement": "Semantic labels are assigned only to reconstructed 3D points whose source pixels are inside 2D masks, so semantics are a subset of the VGGT geometry rather than a separate coordinate system.",
        "coordinate_frame": "semantic_pointcloud.ply uses the same first-camera aligned coordinate frame as pointcloud.ply and scene.glb; semantic_labeled_points.npz also stores raw VGGT world coordinates.",
        "frame_indices": frame_indices,
        "num_semantic_points": int(len(points)),
        "source_depth_conf_median": float(np.median(sources["depth_conf"])) if len(points) else None,
        "classes": class_payload,
        "summary": summary,
        "detections_by_frame": detections_by_frame,
        "artifacts": {
            "semantic_pointcloud_ply": str(ply_path),
            "semantic_pointcloud_with_labels_ply": str(labeled_ply_path),
            "semantic_pointcloud_with_corner_legend_ply": str(corner_legend_ply_path),
            "semantic_labeled_points_npz": str(semantic_dir / "semantic_labeled_points.npz"),
            "semantic_overlays": str(overlay_path),
            "semantic_legend": str(legend_path),
            "semantic_labeled_viewer": str(html_path),
        },
    }
    (semantic_dir / "semantic_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] semantic points: {len(points)}", flush=True)
    print(f"[done] {ply_path}", flush=True)
    print(f"[done] {labeled_ply_path}", flush=True)
    print(f"[done] {corner_legend_ply_path}", flush=True)
    print(f"[done] {overlay_path}", flush=True)
    print(f"[done] {html_path}", flush=True)


if __name__ == "__main__":
    main()
