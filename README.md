# CCO Flight Planner

Generate **DJI CCO (Circular-Coverage Orbit)** waylines from a KML **Polygon** — with an interactive preview GUI, automatic splitting for large missions, and optional KMZ packaging.

> Script: `create_wpml_kml_batch.py`

![status](https://img.shields.io/badge/status-stable-blue) ![python](https://img.shields.io/badge/Python-3.8%2B-informational) ![license](https://img.shields.io/badge/License-Choose--one-lightgrey)

---

##  Features

- **Cover-style CCO route generation**: grid of equal-radius circular passes that cover a polygon region
- **Interactive GUI** (TkAgg / QtAgg + matplotlib): live slider tuning, keyboard fine control, open-new-KML, _Apply & Generate_
- **Autospacing**: center spacing auto-derives from radius & overlap (or set explicitly)
- **Pruning**: when not clipping to polygon, drop circles that don’t touch the polygon interior
- **Outputs**: `template.kml`, `waylines.wpml`, and optional packaged `cco_full.kmz`
- **Slicing**: split into parts when waypoints exceed a chosen threshold; optionally pack each part into `partXX.kmz`
- **Device tags**: embeds DJI `droneInfo` / `payloadInfo` into KML/WPML (e.g., Matrice 4T + H20T gimbal indices)
- **Static preview PNG** (no GUI needed in CLI fallback)

---

##  Requirements

- Python **3.8+**
- Packages:
  - `matplotlib` (for preview/GUI; static PNG also uses it)
  - (Optional GUI backends) **Tk ≥ 8.6** or **PyQt5 / PySide6**

Install:
```bash
pip install matplotlib
# If you prefer Qt:
pip install PyQt5    # or: pip install PySide6
```

> The script gracefully falls back to a **text-based CLI** when GUI backends are unavailable.

---

##  Quick Start

### A) One‑click Standalone (GUI)
Double‑click `create_wpml_kml_batch.py` or run without arguments:
```bash
python create_wpml_kml_batch.py
```
Flow:
1. Select a **Polygon KML**.
2. Choose an **output folder**.
3. Adjust sliders / device fields → **Apply & Generate**.
4. Outputs are written to the chosen folder (and a static `preview.png` is saved).

### B) Command Line (Batch)
```bash
python create_wpml_kml_batch.py \
  --polygon ./demo.kml \
  --drone_enum 99 --drone_sub 1 \
  --payload_enum 89 --payload_sub 0 --payload_pos 0 \
  --circle_radius 30 --circle_overlap 0.3 --per_ring 60 \
  --alt 50 --gimbal -45 --speed 6 \
  --out_dir ./output --max_points 300 \
  --grid_bearing 0 --clip_inside 1 \
  --preview_gui 0
```

Windows (PowerShell) example:
```powershell
python .\create_wpml_kml_batch.py `
  --polygon .\demo.kml `
  --drone_enum 99 --drone_sub 1 `
  --payload_enum 89 --payload_sub 0 --payload_pos 0 `
  --circle_radius 30 --circle_overlap 0.3 --per_ring 60 `
  --alt 50 --gimbal -45 --speed 6 `
  --out_dir .\output --max_points 300 `
  --grid_bearing 0 --clip_inside 1 `
  --preview_gui 0
```

---

##  Inputs & Outputs

**Input**
- A KML file containing a single **`<Polygon>`** (outer boundary).

**Outputs (in `--out_dir`)**
- `template.kml` – KML template with waypoints + embedded device/mission config
- `waylines.wpml` – DJI waylines file
- `cco_full.kmz` – (optional) packaged WPMZ structure with both files
- `parts/` – (optional) split `template_partXX.kml` + `template_partXX.wpml` and packaged `partXX.kmz`
- `preview.png` – static plot of polygon, centers, and path (saved in GUI & CLI fallback)

---

##  CLI Options

| Option | Type / Default | Description |
|---|---|---|
| `--polygon` | **required** | Path to KML containing a `Polygon`. |
| `--center_mode` | `centroid` \| `bbox_center` (default: `centroid`) | Choose center reference for grid build. |
| `--per_ring` | int (default **60**, min 3) | Points per circle. |
| `--start_bearing` | float (deg, default **0.0**) | Start angle (0 = North, clockwise positive). |
| `--alt` | float m (default **50.0**) | Waypoint altitude. |
| `--speed` | float m/s (default **6.0**) | Mission transitional speed. |
| `--gimbal` | float deg (default **-45.0**) | Gimbal pitch (negative = down). |
| `--file_suffix` | str (default **LiangchaoDeng_SHZU**) | Appended to photo filenames via action group. |
| `--kmz` | int 0/1 (default **1**) | Whether to package a full KMZ. |
| `--out_dir` | str (default **output**) | Output directory. |
| `--max_points` | int (default **300**) | Split threshold for sub-missions. |
| `--pack_full_kmz` | int 0/1 (default **1**) | Pack `cco_full.kmz`. |
| `--pack_parts_kmz` | int 0/1 (default **1**) | Pack `partXX.kmz` in `parts/`. |
| `--preview` | int 0/1 (default **0**) | Show a non-interactive preview window. |
| **Cover mode** |||
| `--circle_radius` | float m (default **30.0**) | Circle radius. |
| `--center_step` | float m (default **0.0**) | Center spacing; `0` = auto from radius & overlap. |
| `--circle_overlap` | 0~0.9 (default **0.3**) | Linear overlap between adjacent circles. |
| `--cover_padding` | float m (default **10.0**) | Expand polygon bbox to reduce edge gaps. |
| `--grid_bearing` | float deg (default **0.0**) | Grid rotation (clockwise). |
| `--clip_inside` | int 0/1 (default **1**) | Keep only waypoints inside polygon. |
| `--prune_outside` | int 0/1 (default **1**) | When `clip_inside=0`, drop centers with no interior points. |
| **GUI** |||
| `--preview_gui` | int 0/1 (default **0**) | Launch interactive GUI before generation. |
| **Device tags** |||
| `--drone_enum` | int / optional | DJI `droneEnumValue`. Example **99** (Matrice 4T). |
| `--drone_sub` | int / optional | DJI `droneSubEnumValue`. Example **1**. |
| `--payload_enum` | int / optional | DJI `payloadEnumValue`. Example **89**. |
| `--payload_sub` | int / optional | DJI `payloadSubEnumValue`. Example **0**. |
| `--payload_pos` | int / optional | DJI `payloadPositionIndex`. Example **0**. |

> **Tip**: If `--center_step` is `0`, effective spacing is `2 * radius * (1 - overlap)` (clamped to ≥ 1 m).

---

##  GUI Controls & Shortcuts

- **Sliders** (right panel): `Radius`, `Pts/circ`, `Overlap`, `Center step`, `Padding`, `Bearing`, `Max points`
- **TextBoxes**: `DroneEnum`, `DroneSub`, `PayloadEnum`, `PayloadSub`, `PayloadPos`
- **Buttons**: `Clip ON/OFF`, `Prune ON/OFF`, `Open KML…`, `Apply & Generate`
- **Keyboard**:
  - `Tab` / `Down` — select next slider; `Shift+Tab` / `Up` — previous
  - `Left` / `Right` — fine-step active slider; **hold Shift** for ×10 step

---

##  How It Works (Quick Notes)

- Builds a **grid of circle centers** over an expanded polygon bounding box (with optional rotation), then **snake-orders rows**.
- For each center, sample `per_ring` points on the circle. If **Clip = ON**, keep only points inside polygon.
- **Start angles** on each circle are **realigned** to minimize jumps between neighboring circles; ring direction alternates.
- When **clip = OFF** and **prune = ON**, circles with no interior samples are dropped to save time/space.

---

##  Troubleshooting

- **No GUI pops up**
  - Ensure **Tk ≥ 8.6** (`python -m tkinter`) or install **PyQt5 / PySide6**.
  - Headless servers: use CLI; static `preview.png` still works.
- **KMZ won’t import**
  - DJI apps sometimes expect specific WPMZ structure: this tool writes `wpmz/template.kml` and `wpmz/waylines.wpml` at the archive root.
- **Waypoints exceed device limits**
  - Use a smaller `per_ring` or larger `center_step` / `radius`, or lower `--max_points` to enable slicing.
- **Device fields**
  - If you don’t know exact enums, leave them empty; files still generate without embedded device tags.

---

##  Project Layout

```
create_wpml_kml_batch.py   # main script
README.md                  # this file
```

You may later add:
```
examples/
  demo.kml
  screenshots/preview.png
```

---

##  Contributing

Issues and PRs are welcome. Please include a minimal KML polygon to reproduce any problem reports.

---

##  License

Choose an OSI-approved license (e.g., MIT, Apache-2.0). Add it as `LICENSE` in the repo root and update the badge at the top.

---

##  Citation (optional)

If this tool helps your research, consider citing it:

```
@software{cco_flight_planner,
  title   = {CCO Flight Planner: Generate DJI CCO waylines from KML Polygons},
  author  = {Your Name},
  year    = {2025},
  url     = {https://github.com/<yourname>/cco-flight-planner}
}
```

---

## ✍ Acknowledgments

- Built for field missions that require **dense circular coverage** over irregular plots.
- Defaults tuned for **Matrice 4T** class hardware (`drone_enum=99`, `drone_sub=1`, `payload_enum=89`, `payload_sub=0`, `payload_pos=0`). Adapt as needed.
