"""Smoke tests: generate the demo mission, split it and check the outputs."""
import os
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import create_wpml_kml_batch as planner  # noqa: E402

NS = planner.NS
DEMO = ROOT / "examples" / "demo_field.kml"


def test_polygon_and_centroid():
    poly = planner.parse_polygon_kml(DEMO)
    assert len(poly) == 4
    lon, lat = planner.polygon_centroid(poly)
    assert abs(lon - 86.05075) < 1e-6 and abs(lat - 44.30035) < 1e-6


def test_polygon_without_namespace(tmp_path):
    text = DEMO.read_text(encoding="utf-8").replace(' xmlns="http://www.opengis.net/kml/2.2"', "")
    f = tmp_path / "plain.kml"
    f.write_text(text, encoding="utf-8")
    assert len(planner.parse_polygon_kml(f)) == 4


def test_waypoints_inside_and_facing_centre():
    poly = planner.parse_polygon_kml(DEMO)
    lon_c, lat_c = planner.polygon_centroid(poly)
    cfg = dict(planner.DEFAULTS, circle_radius=20, per_ring=24)
    centers, pts = planner.build_waypoints(poly, lon_c, lat_c, cfg)
    assert centers and pts
    assert all(planner.point_in_polygon(lon, lat, poly) for lon, lat, _ in pts)
    assert all(0 <= h < 360 for _, _, h in pts)


def test_cli_outputs_and_split(tmp_path):
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "create_wpml_kml_batch.py"),
                    "--polygon", str(DEMO), "--circle_radius", "20", "--per_ring", "24",
                    "--max_points", "100", "--out_dir", str(out), "--file_suffix", "test",
                    "--drone_enum", "99", "--drone_sub", "1",
                    "--payload_enum", "89", "--payload_sub", "0", "--payload_pos", "0"],
                   check=True, capture_output=True)
    for name in ("template.kml", "waylines.wpml", "cco_full.kmz", "preview.png"):
        assert (out / name).exists(), name
    names = zipfile.ZipFile(out / "cco_full.kmz").namelist()
    assert {"wpmz/template.kml", "wpmz/waylines.wpml", "wpmz/res/"} <= set(names)
    parts = sorted((out / "parts").glob("part*.kmz"))
    assert len(parts) >= 2
    tpl = (out / "template.kml").read_text(encoding="utf-8")
    assert "droneEnumValue>99<" in tpl and "fileSuffix>test<" in tpl

    # the splitter keeps device tags and renumbers waypoints
    split = tmp_path / "split"
    subprocess.run([sys.executable, str(ROOT / "split_wpml_kml_batch.py"), "--max", "50",
                    "--out_dir", str(split), "--inputs",
                    str(out / "template.kml"), str(out / "waylines.wpml")],
                   check=True, capture_output=True)
    first = ET.parse(split / "template_part01.kml").getroot()
    assert first.find(".//wpml:droneInfo", NS) is not None
    second = ET.parse(split / "waylines_part02.wpml").getroot()
    idx = [int(e.text) for e in second.iter(f"{{{NS['wpml']}}}index")]
    assert idx[:3] == [0, 1, 2]
