# split_wpml_kml_batch.py
# 用法示例：
#   python split_wpml_kml_batch.py ^
#     --max 300 ^
#     --out_dir .\splits ^
#     --inputs .\main_route.kml .\companion_route.kml

import os
import argparse
import xml.etree.ElementTree as ET
from copy import deepcopy
import shutil
import zipfile

NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "wpml": "http://www.dji.com/wpmz/1.0.4",
}
# 注册命名空间，写回时保持前缀
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
        raise ValueError("未找到任何 Placemark。")
    return doc, folder, placemarks

def clone_folder_without_placemarks(folder):
    nf = ET.Element(folder.tag, folder.attrib)
    for ch in list(folder):
        if ch.tag.endswith("Placemark"):
            continue
        nf.append(deepcopy(ch))
    return nf

def reindex_one_placemark(pm, new_idx):
    # <wpml:index>
    idx = pm.find("wpml:index", NS)
    if idx is None:
        idx = ET.SubElement(pm, f"{{{NS['wpml']}}}index")
    idx.text = str(new_idx)

    # actionGroup（常见为到点拍照一组动作）
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
    new_doc = ET.Element(orig_doc.tag, orig_doc.attrib)

    mc = orig_doc.find("wpml:missionConfig", NS)
    if mc is not None:
        new_doc.append(deepcopy(mc))

    nf = clone_folder_without_placemarks(folder_template)
    # 可选：把 templateId 写成分片号，便于区分
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
    total = len(pms)

    base = os.path.splitext(os.path.basename(in_path))[0]
    for pi, (s, e) in enumerate(cuts):
        new_root = ET.Element(root.tag, root.attrib)
        new_doc = build_doc_slice(doc, folder, pms[s:e], pi if part_no_from_zero else pi+1, renumber=True)
        new_root.append(new_doc)
        ext = os.path.splitext(in_path)[1] or ".kml"
        out_file = os.path.join(out_dir, f"{base}_part{pi+1:02d}{ext}")
        ET.ElementTree(new_root).write(out_file, encoding="utf-8", xml_declaration=True)
        print(f"[{base}] -> {out_file} (航点 {s}-{e-1}, 共 {e-s} 个)")

def guess_role_for_input(path):
    """根据文件名/扩展名推断角色：'template' 或 'waylines'。"""
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1].lower()
    if "waylines" in name or ext == ".wpml":
        return "waylines"
    # 默认把 .kml 当 template
    return "template"

def build_kmz_for_part(out_dir, inputs_info, part_index):
    """
    将某一分片的文件打包为 KMZ：
    结构：
      partXX.kmz
        └─ wpmz/
           ├─ template.kml
           ├─ waylines.wpml
           └─ res/   (空目录)
    inputs_info: [{'base':..., 'ext':..., 'role': 'template'|'waylines'}]
    part_index: 从 1 开始
    """
    # 准备临时构建目录
    part_tag = f"part{part_index:02d}"
    build_root = os.path.join(out_dir, f"{part_tag}_build")
    wpmz_dir = os.path.join(build_root, "wpmz")
    res_dir = os.path.join(wpmz_dir, "res")
    os.makedirs(res_dir, exist_ok=True)

    # 复制并重命名为标准文件名
    got_template = False
    got_waylines = False
    for info in inputs_info:
        src = os.path.join(out_dir, f"{info['base']}_part{part_index:02d}{info['ext']}")
        if not os.path.exists(src):
            raise FileNotFoundError(f"未找到分片文件：{src}")
        if info["role"] == "waylines":
            dst = os.path.join(wpmz_dir, "waylines.wpml")
            shutil.copyfile(src, dst)
            got_waylines = True
        else:
            # 默认视为 template
            dst = os.path.join(wpmz_dir, "template.kml")
            shutil.copyfile(src, dst)
            got_template = True

    # 生成 kmz（zip）
    kmz_path = os.path.join(out_dir, f"{part_tag}.kmz")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 写入目录占位（可选）
        zf.writestr("wpmz/", "")
        zf.writestr("wpmz/res/", "")
        # 写入文件
        if got_template:
            zf.write(os.path.join(wpmz_dir, "template.kml"), arcname="wpmz/template.kml")
        if got_waylines:
            zf.write(os.path.join(wpmz_dir, "waylines.wpml"), arcname="wpmz/waylines.wpml")

    # 清理临时目录
    shutil.rmtree(build_root, ignore_errors=True)
    print(f"[KMZ] -> {kmz_path}  (已包含 wpmz/template.kml, wpmz/waylines.wpml, 空 res/)")

def main():
    ap = argparse.ArgumentParser(description="按首个 KML 的航点切片方案，对多个 WPML KML 同步分割")
    ap.add_argument("--inputs", nargs="*", required=False, help="多个输入 KML，首个为切片基准；若省略则自动寻找同目录 template.kml 与 waylines.wpml")
    ap.add_argument("--max", type=int, default=300, help="单文件最大航点数，默认 300")
    ap.add_argument("--out_dir", default="splits", help="输出目录，默认 ./splits")
    args = ap.parse_args()

    # 若未提供 --inputs，则自动使用脚本同目录下的 template.kml 和 waylines.wpml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.inputs:
        auto_template = os.path.join(script_dir, "template.kml")
        auto_waylines = os.path.join(script_dir, "waylines.wpml")
        if not os.path.exists(auto_template):
            raise FileNotFoundError(f"未提供 --inputs，且未找到默认文件：{auto_template}")
        if not os.path.exists(auto_waylines):
            raise FileNotFoundError(f"未提供 --inputs，且未找到默认文件：{auto_waylines}")
        args.inputs = [auto_template, auto_waylines]
        print(f"未提供 --inputs，自动使用: {args.inputs}")

    # 输出目录若为相对路径，则放到脚本同目录下
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(script_dir, args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # 以第一个 KML 为基准计算切片
    base_tree = ET.parse(args.inputs[0])
    base_root = base_tree.getroot()
    _, _, base_pms = get_doc_folder_placemarks(base_root)
    total = len(base_pms)
    cuts = split_points(total, args.max)
    print(f"基准文件共有 {total} 个航点，将切成 {len(cuts)} 份。切片区间：{cuts}")

    # 逐个文件同步分割（校验航点数一致）
    for p in args.inputs:
        t = ET.parse(p)
        r = t.getroot()
        _, _, pms = get_doc_folder_placemarks(r)
        if len(pms) != total:
            raise ValueError(f"文件 {p} 的航点数 {len(pms)} 与基准 {total} 不一致，无法联动切分。")
        process_one_file(p, cuts, args.out_dir)

    # ------------------------------------------------------------
    # 组装 KMZ：将每个分片的配对文件打包为 partXX.kmz
    # ------------------------------------------------------------
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