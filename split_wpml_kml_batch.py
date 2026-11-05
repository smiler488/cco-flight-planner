"""
Split existing DJI WPML missions into sub-missions of at most N waypoints.

Use this for routes that already exist (made in DJI Pilot 2, DJI Terra or by
create_wpml_kml_batch.py). All input files are cut at the same waypoint
indices, so a template.kml and its waylines.wpml stay paired, and each part
is packed as partXX.kmz ready for import.

Example:
    python split_wpml_kml_batch.py --max 300 --out_dir splits \
        --inputs output/template.kml output/waylines.wpml

Without --inputs, template.kml and waylines.wpml in the current folder are used.

Author: Liangchao Deng (邓良超)
License: MIT
"""

import os
import argparse
import xml.etree.ElementTree as ET
from copy import deepcopy
import zipfile

NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "wpml": "http://www.dji.com/wpmz/1.0.4",
}
# keep the original prefixes when writing back
for prefix, uri in NS.items():
    ET.register_namespace(prefix if prefix != "kml" else "", uri)

def fone(p, path):
    el = p.find(path, NS)
    if el is None:
        raise ValueError(f"Missing required element: {path}")
    return el

def fall(p, path):
    return p.findall(path, NS)

def get_doc_folder_placemarks(root):
    doc = fone(root, "kml:Document")
    folder = fone(doc, "kml:Folder")
    placemarks = fall(folder, "kml:Placemark")
    if not placemarks:
        raise ValueError("No Placemark found in the first Folder.")
    return doc, folder, placemarks

def clone_folder_without_placemarks(folder):
    nf = ET.Element(folder.tag, folder.attrib)
    for ch in list(folder):
        if ch.tag.endswith("Placemark"):
            continue
        nf.append(deepcopy(ch))
    return nf

def reindex_one_placemark(pm, new_idx):
    idx = pm.find("wpml:index", NS)
    if idx is None:
        idx = ET.SubElement(pm, f"{{{NS['wpml']}}}index")
    idx.text = str(new_idx)

    # actionGroup ids follow the waypoint index (one take-photo group per waypoint)
    ag = pm.find("wpml:actionGroup", NS)
    if ag is not None:
        agid = ag.find("wpml:actionGroupId", NS)
        if agid is None:
            agid = ET.SubElement(ag, f"{{{NS['wpml']}}}actionGroupId")
        agid.text = str(new_idx)

        ags = ag.find("wpml:actionGroupStartIndex", NS)
        if ags is None:
            ags = ET.SubElement(ag, f"{{{NS['wpml']}}}actionGroupStartIndex")
        ags.text = str(new_idx)

        age = ag.find("wpml:actionGroupEndIndex", NS)
        if age is None:
            age = ET.SubElement(ag, f"{{{NS['wpml']}}}actionGroupEndIndex")
        age.text = str(new_idx)
    return pm

def build_doc_slice(orig_doc, folder_template, placemark_slice, part_id, renumber=True):
    """New Document with every non-Folder child of the original (missionConfig,
    droneInfo, payloadInfo, author, ...) and one Folder holding the slice."""
    new_doc = ET.Element(orig_doc.tag, orig_doc.attrib)
    for child in orig_doc:
        if child.tag != folder_template.tag:
            new_doc.append(deepcopy(child))

    nf = clone_folder_without_placemarks(folder_template)
    # templateId = part number, so the parts can be told apart on the controller
    tid = nf.find("wpml:templateId", NS)
    if tid is None:
        tid = ET.SubElement(nf, f"{{{NS['wpml']}}}templateId")
    tid.text = str(part_id)

    for i, pm in enumerate(placemark_slice):
        cpm = deepcopy(pm)
        if renumber:
            cpm = reindex_one_placemark(cpm, i)
        nf.append(cpm)

    new_doc.append(nf)
    return new_doc

def split_points(total, max_points):
    cuts = []
    for s in range(0, total, max_points):
        e = min(s + max_points, total)
        cuts.append((s, e))
    return cuts

def process_one_file(in_path, cuts, out_dir, part_no_from_zero=True):
    tree = ET.parse(in_path)
    root = tree.getroot()
    doc, folder, pms = get_doc_folder_placemarks(root)

    base = os.path.splitext(os.path.basename(in_path))[0]
    for pi, (s, e) in enumerate(cuts):
        new_root = ET.Element(root.tag, root.attrib)
        new_doc = build_doc_slice(doc, folder, pms[s:e], pi if part_no_from_zero else pi+1, renumber=True)
        new_root.append(new_doc)
        ext = os.path.splitext(in_path)[1] or ".kml"
        out_file = os.path.join(out_dir, f"{base}_part{pi+1:02d}{ext}")
        ET.ElementTree(new_root).write(out_file, encoding="utf-8", xml_declaration=True)
        print(f"[{base}] -> {out_file} (waypoints {s}-{e-1}, {e-s} in total)")

def guess_role_for_input(path):
    """'waylines' for *.wpml or names containing 'waylines', otherwise 'template'."""
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1].lower()
    if "waylines" in name or ext == ".wpml":
        return "waylines"
    return "template"

def build_kmz_for_part(out_dir, inputs_info, part_index):
    """Pack part ``part_index`` (1-based) as partXX.kmz with the WPMZ layout
    wpmz/template.kml, wpmz/waylines.wpml and an empty wpmz/res/.

    inputs_info: [{'base': ..., 'ext': ..., 'role': 'template' | 'waylines'}]
    """
    part_tag = f"part{part_index:02d}"
    kmz_path = os.path.join(out_dir, f"{part_tag}.kmz")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wpmz/", "")
        zf.writestr("wpmz/res/", "")
        for info in inputs_info:
            src = os.path.join(out_dir, f"{info['base']}_part{part_index:02d}{info['ext']}")
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing part file: {src}")
            name = "waylines.wpml" if info["role"] == "waylines" else "template.kml"
            zf.write(src, arcname=f"wpmz/{name}")
    print(f"[KMZ] -> {kmz_path}")

def main():
    ap = argparse.ArgumentParser(description="Split paired WPML files (template.kml / waylines.wpml) at the same waypoint indices.")
    ap.add_argument("--inputs", nargs="*", help="input files; the first one defines the cut points "
                                                "(default: template.kml and waylines.wpml in the current folder)")
    ap.add_argument("--max", type=int, default=300, help="max waypoints per part (default 300)")
    ap.add_argument("--out_dir", default="splits", help="output folder (default ./splits)")
    ap.add_argument("--no_kmz", action="store_true", help="only write the split files, no partXX.kmz")
    args = ap.parse_args()
    if args.max < 1:
        ap.error("--max must be at least 1")

    if not args.inputs:
        args.inputs = ["template.kml", "waylines.wpml"]
        for f in args.inputs:
            if not os.path.exists(f):
                ap.error(f"no --inputs given and {f} not found in the current folder")
        print(f"Using {args.inputs}")

    os.makedirs(args.out_dir, exist_ok=True)

    # The first file defines the cut points
    base_tree = ET.parse(args.inputs[0])
    base_root = base_tree.getroot()
    _, _, base_pms = get_doc_folder_placemarks(base_root)
    total = len(base_pms)
    cuts = split_points(total, args.max)
    print(f"{total} waypoints -> {len(cuts)} parts: {cuts}")

    # Every file must have the same number of waypoints
    for p in args.inputs:
        t = ET.parse(p)
        r = t.getroot()
        _, _, pms = get_doc_folder_placemarks(r)
        if len(pms) != total:
            raise ValueError(f"{p} has {len(pms)} waypoints but {args.inputs[0]} has {total}; cannot split them together.")
        process_one_file(p, cuts, args.out_dir)

    if args.no_kmz:
        return

    # Pack the paired files of each part as partXX.kmz
    inputs_info = []
    for p in args.inputs:
        base = os.path.splitext(os.path.basename(p))[0]
        ext = os.path.splitext(p)[1] or ".kml"
        role = guess_role_for_input(p)
        inputs_info.append({"base": base, "ext": ext, "role": role})

    parts_count = len(cuts)
    for pi in range(1, parts_count + 1):
        build_kmz_for_part(args.out_dir, inputs_info, pi)

if __name__ == "__main__":
    main()