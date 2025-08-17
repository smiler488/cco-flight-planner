# generate_cco_from_polygon.py  (cover, preview, split, KMZ, standalone GUI)
# 功能：读取包含 <Polygon> 的 KML，生成覆盖式 CCO 航线（template.kml + waylines.wpml + 可选 KMZ）
# 典型用法（命令行覆盖式）：
#   python create_wpml_kml_batch.py \
#     --polygon ./2025sj.kml \
#     --drone_enum 99 --drone_sub 1 \
#     --payload_enum 89 --payload_sub 0 --payload_pos 0 \
#     --circle_radius 5 --circle_overlap 0.3 --per_ring 18 \
#     --alt 5 --gimbal -45 --speed 6 \
#     --out_dir ./output --max_points 300 \
#     --grid_bearing 0 --clip_inside 1 \
#     --preview_gui 0
#
# 直接双击/无参数运行：进入 Standalone GUI 流程（选择 KML 与输出目录、滑块实时预览、Apply & Generate 后落盘）

import os, math, argparse, xml.etree.ElementTree as ET, zipfile, shutil, sys
# Backend selection: prefer TkAgg if Tk>=8.6; else try QtAgg (PyQt5/PySide6); else fall back to Agg
TK_OK = False
USE_QT = False
try:
    import tkinter as _tk
    TK_OK = float(getattr(_tk, "TkVersion", 0)) >= 8.6
except Exception:
    TK_OK = False

# Probe Qt bindings
_qt_mod = None
try:
    import PyQt5  # noqa: F401
    _qt_mod = "PyQt5"
    USE_QT = True
except Exception:
    try:
        import PySide6  # noqa: F401
        _qt_mod = "PySide6"
        USE_QT = True
    except Exception:
        USE_QT = False

try:
    import matplotlib
    if os.environ.get("MPLBACKEND", "") == "":
        if TK_OK:
            matplotlib.use("TkAgg")
        elif USE_QT:
            # Matplotlib 3.8+ uses 'QtAgg'; older versions accept 'Qt5Agg'
            try:
                matplotlib.use("QtAgg")
            except Exception:
                matplotlib.use("Qt5Agg")
        else:
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, TextBox
except Exception:
    plt = None
    Slider = None
    Button = None
    TextBox = None

# File dialog adapters (Tk or Qt)
DIALOG_BACKEND = None
Tk = None; filedialog = None
if TK_OK:
    try:
        from tkinter import Tk, filedialog  # type: ignore
        DIALOG_BACKEND = "tk"
    except Exception:
        Tk = None; filedialog = None

if DIALOG_BACKEND is None and USE_QT:
    try:
        # Lazy import Qt only for dialogs
        if _qt_mod == "PyQt5":
            from PyQt5 import QtWidgets  # type: ignore
        else:
            from PySide6 import QtWidgets  # type: ignore
        DIALOG_BACKEND = "qt"
    except Exception:
        DIALOG_BACKEND = None

def pick_open_file_kml():
    """Open a file dialog to pick a KML path. Returns str or '' if canceled."""
    if DIALOG_BACKEND == "tk":
        try:
            root = Tk(); root.withdraw()
            path = filedialog.askopenfilename(title="Select Polygon KML", filetypes=[("KML files", "*.kml"), ("All files", "*.*")])
            root.update(); root.destroy()
            return path or ""
        except Exception:
            return ""
    if DIALOG_BACKEND == "qt":
        try:
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Polygon KML", "", "KML files (*.kml);;All files (*)")
            return path or ""
        except Exception:
            return ""
    return ""

def pick_select_folder():
    """Open a directory chooser. Returns str or '' if canceled."""
    if DIALOG_BACKEND == "tk":
        try:
            root = Tk(); root.withdraw()
            path = filedialog.askdirectory(title="Select output folder")
            root.update(); root.destroy()
            return path or ""
        except Exception:
            return ""
    if DIALOG_BACKEND == "qt":
        try:
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            dlg = QtWidgets.QFileDialog(None, "Select output folder")
            dlg.setFileMode(QtWidgets.QFileDialog.Directory)
            dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
            if dlg.exec_():
                paths = dlg.selectedFiles()
                return (paths[0] if paths else "") or ""
            return ""
        except Exception:
            return ""
    return ""

from copy import deepcopy

# ------------------ 命名空间 ------------------
NS = {
    "kml":  "http://www.opengis.net/kml/2.2",
    "wpml": "http://www.dji.com/wpmz/1.0.4",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix if prefix != "kml" else "", uri)

# ------------------ 设备信息（飞机/载荷）写入 ------------------
def add_device_info_to_doc(doc, device):
    if not device:
        return
    d_ev = device.get("drone_enum", None) if isinstance(device, dict) else None
    d_sev = device.get("drone_sub", None) if isinstance(device, dict) else None
    p_ev = device.get("payload_enum", None) if isinstance(device, dict) else None
    p_sev = device.get("payload_sub", None) if isinstance(device, dict) else None
    p_pos = device.get("payload_pos", None) if isinstance(device, dict) else None
    try:
        if d_ev is not None and d_sev is not None:
            dinfo = el("droneInfo")
            dinfo.append(el("droneEnumValue", int(d_ev)))
            dinfo.append(el("droneSubEnumValue", int(d_sev)))
            doc.append(dinfo)
    except Exception:
        pass
    try:
        if p_ev is not None and p_sev is not None and p_pos is not None:
            pinfo = el("payloadInfo")
            pinfo.append(el("payloadEnumValue", int(p_ev)))
            pinfo.append(el("payloadSubEnumValue", int(p_sev)))
            pinfo.append(el("payloadPositionIndex", int(p_pos)))
            doc.append(pinfo)
    except Exception:
        pass

# ------------------ 基本几何 ------------------
def meters_to_deg(lat_deg, dx_m, dy_m):
    """米->经纬度偏移（近似，够用）：dx 向东(+)、dy 向北(+)"""
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat_rad) + 1.175*math.cos(4*lat_rad)
    m_per_deg_lon = 111412.84*math.cos(lat_rad) - 93.5*math.cos(3*lat_rad)
    dlat = dy_m / m_per_deg_lat
    dlon = dx_m / m_per_deg_lon
    return dlon, dlat

def bearing_to_vector(bearing_deg, dist_m):
    """方位角(0=北,顺时针) + 距离 -> 平面位移(dx,dy)"""
    b = math.radians(bearing_deg)
    dx = dist_m * math.sin(b)  # 东
    dy = dist_m * math.cos(b)  # 北
    return dx, dy

def normalize_heading(deg):
    x = deg % 360.0
    return x if x >= 0 else x+360

# 旋转与点在多边形内判断
def rotate_xy(x, y, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return (x*ca - y*sa, x*sa + y*ca)

def point_in_polygon(lon, lat, poly):
    # ray casting, poly: [(lon,lat), ...] without duplicate last
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        cond = ((y1 > lat) != (y2 > lat))
        if cond:
            xinters = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
            if lon < xinters:
                inside = not inside
    return inside

# --- Helper: check if a circle (sampled) touches the polygon (at least one waypoint inside) ---
def circle_touches_polygon(center_lon, center_lat, radius_m, per_circle, start_bearing_deg, poly):
    """Return True if at least one sampled point on the circle lies inside the polygon.
    Uses same sampling density as per_circle to be consistent with generation."""
    angle_step = 360.0 / max(per_circle, 3)
    for k in range(max(per_circle, 3)):
        ang = start_bearing_deg + k * angle_step
        dx, dy = bearing_to_vector(ang, radius_m)
        dlon, dlat = meters_to_deg(center_lat, dx, dy)
        lon = center_lon + dlon
        lat = center_lat + dlat
        if point_in_polygon(lon, lat, poly):
            return True
    return False

# ------------------ 多边形解析 ------------------
def parse_polygon_kml(path):
    """返回 [(lon,lat), ...] 外环坐标（假设单 Polygon）"""
    tree = ET.parse(path)
    root = tree.getroot()
    doc = root.find("kml:Document", NS) or root
    coords_text = None
    for placemark in doc.findall(".//kml:Placemark", NS):
        poly = placemark.find(".//kml:Polygon", NS)
        if poly is None:
            continue
        ring = poly.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", NS)
        if ring is not None and ring.text and ring.text.strip():
            coords_text = ring.text
            break
    if not coords_text:
        raise ValueError("未在 KML 中找到 Polygon 坐标。")

    pts = []
    for line in coords_text.strip().split():
        sp = line.split(",")
        if len(sp) >= 2:
            lon = float(sp[0]); lat = float(sp[1])
            pts.append((lon, lat))
    if len(pts) > 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("Polygon 顶点不足（<3）。")
    return pts

def polygon_centroid(lon_lat_pts):
    """多边形质心（经纬度）——用平面近似（对小地块足够）"""
    lon0, lat0 = lon_lat_pts[0]
    lat_rad = math.radians(lat0)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat_rad) + 1.175*math.cos(4*lat_rad)
    m_per_deg_lon = 111412.84*math.cos(lat_rad) - 93.5*math.cos(3*lat_rad)

    xy = []
    for lon, lat in lon_lat_pts:
        x = (lon - lon0) * m_per_deg_lon
        y = (lat - lat0) * m_per_deg_lat
        xy.append((x, y))
    A = 0.0; Cx = 0.0; Cy = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i+1) % len(xy)]
        cross = x1*y2 - x2*y1
        A  += cross
        Cx += (x1 + x2)*cross
        Cy += (y1 + y2)*cross
    A *= 0.5
    if abs(A) < 1e-9:
        lon_c = sum(p[0] for p in lon_lat_pts)/len(lon_lat_pts)
        lat_c = sum(p[1] for p in lon_lat_pts)/len(lon_lat_pts)
        return lon_c, lat_c
    Cx /= (6*A); Cy /= (6*A)
    lon_c = lon0 + Cx / m_per_deg_lon
    lat_c = lat0 + Cy / m_per_deg_lat
    return lon_c, lat_c

# ------------------ 环间顺序平滑 & 分片（保留距离计算与分片） ------------------
def _m_per_deg(lat_deg):
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat_rad) + 1.175*math.cos(4*lat_rad)
    m_per_deg_lon = 111412.84*math.cos(lat_rad) - 93.5*math.cos(3*lat_rad)
    return m_per_deg_lon, m_per_deg_lat

def _dist_m(p1, p2):
    """两经纬点近似距离（m）"""
    lon1, lat1, _ = p1
    lon2, lat2, _ = p2
    m_per_deg_lon, m_per_deg_lat = _m_per_deg((lat1+lat2)/2.0)
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    return (dx*dx + dy*dy) ** 0.5

def chunk_slices(total, max_points):
    cuts = []
    for s in range(0, total, max_points):
        cuts.append((s, min(s + max_points, total)))
    return cuts

def build_kmz_for_part(out_dir, part_tag, template_src, waylines_src):
    build_root = os.path.join(out_dir, f"{part_tag}_build")
    wpmz_dir = os.path.join(build_root, "wpmz")
    res_dir = os.path.join(wpmz_dir, "res")
    os.makedirs(res_dir, exist_ok=True)
    shutil.copyfile(template_src, os.path.join(wpmz_dir, "template.kml"))
    shutil.copyfile(waylines_src, os.path.join(wpmz_dir, "waylines.wpml"))
    kmz_path = os.path.join(out_dir, f"{part_tag}.kmz")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wpmz/", "")
        zf.writestr("wpmz/res/", "")
        zf.write(os.path.join(wpmz_dir, "template.kml"), arcname="wpmz/template.kml")
        zf.write(os.path.join(wpmz_dir, "waylines.wpml"), arcname="wpmz/waylines.wpml")
    shutil.rmtree(build_root, ignore_errors=True)
    print(f"[KMZ] -> {kmz_path}")

# ------------------ 覆盖式 CCO：等半径圆阵 ------------------
def _bbox_lonlat(poly):
    lons = [p[0] for p in poly]; lats = [p[1] for p in poly]
    return (min(lons), min(lats), max(lons), max(lats))

def _m_scale(lat_deg):
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat_rad) + 1.175*math.cos(4*lat_rad)
    m_per_deg_lon = 111412.84*math.cos(lat_rad) - 93.5*math.cos(3*lat_rad)
    return m_per_deg_lon, m_per_deg_lat

def grid_circle_centers(poly, center_lon, center_lat, step_m, padding_m, bearing_deg=0.0):
    """在多边形外接框(外扩 padding)内，生成步距为 step_m 的圆心网格（蛇形路径顺序）。返回[(lon,lat), ...]"""
    xmin, ymin, xmax, ymax = _bbox_lonlat(poly)
    m_per_deg_lon, m_per_deg_lat = _m_scale(center_lat)
    dx = padding_m / m_per_deg_lon
    dy = padding_m / m_per_deg_lat
    xmin -= dx; xmax += dx; ymin -= dy; ymax += dy
    def lonlat_to_xy(lon, lat):
        return ( (lon - center_lon) * m_per_deg_lon, (lat - center_lat) * m_per_deg_lat )
    def xy_to_lonlat(x, y):
        return ( center_lon + x / m_per_deg_lon, center_lat + y / m_per_deg_lat )
    x0, y0 = lonlat_to_xy(xmin, ymin)
    x1, y1 = lonlat_to_xy(xmax, ymax)
    if step_m <= 0:
        step_m = 1.0
    xs = []
    x = math.floor(min(x0,x1) / step_m) * step_m
    x_max = math.ceil(max(x0,x1) / step_m) * step_m
    while x <= x_max + 1e-6:
        xs.append(x); x += step_m
    ys = []
    y = math.floor(min(y0,y1) / step_m) * step_m
    y_max = math.ceil(max(y0,y1) / step_m) * step_m
    while y <= y_max + 1e-6:
        ys.append(y); y += step_m
    centers = []
    reverse = False
    for j, yy in enumerate(ys):
        row_xy = [rotate_xy(xx, yy, bearing_deg) for xx in xs]
        row = [xy_to_lonlat(xx, yy) for (xx, yy) in row_xy]
        if reverse:
            row = list(reversed(row))
        centers.extend(row)
        reverse = not reverse
    return centers

# --- Helper: prune centers whose circle has no waypoint inside polygon ---
def prune_centers_outside(poly, centers_lonlat, circle_radius_m, per_circle, start_bearing_deg=0.0):
    """Keep only those centers whose circle has at least one waypoint inside poly."""
    kept = []
    for (lon_c, lat_c) in centers_lonlat:
        if circle_touches_polygon(lon_c, lat_c, circle_radius_m, per_circle, start_bearing_deg, poly):
            kept.append((lon_c, lat_c))
    return kept

def generate_cover_cco_points(centers_lonlat, per_circle, circle_radius_m, start_bearing_deg=0.0, clip_poly=None):
    """
    对每个圆心生成一圈相同半径的点（per_circle 个），并蛇形拼接：
    - 起始角 = start_bearing_deg
    - 相邻圆之间，为减少跨圆跳距：自动旋转该圆的起点角到距离上一圆最后一个点最近的位置；并交替正/反向
    返回 [(lon,lat,heading), ...]
    """
    if not centers_lonlat:
        return []
    all_points = []
    last_pt = None
    reverse_dir = False
    for (lon_c, lat_c) in centers_lonlat:
        pts_circle = []
        angle_step = 360.0 / max(per_circle, 3)
        for k in range(per_circle):
            ang = start_bearing_deg + k * angle_step
            dx, dy = bearing_to_vector(ang, circle_radius_m)
            dlon, dlat = meters_to_deg(lat_c, dx, dy)
            lat = lat_c + dlat; lon = lon_c + dlon
            head = normalize_heading(math.degrees(math.atan2(-dx, -dy)))  # 机头指向圆心
            if (clip_poly is None) or point_in_polygon(lon, lat, clip_poly):
                pts_circle.append((lon, lat, head))
        seq = list(reversed(pts_circle)) if reverse_dir else pts_circle
        if last_pt is not None:
            best_i, best_d = 0, float('inf')
            for i, p in enumerate(seq):
                d = _dist_m(last_pt, (p[0], p[1], 0.0))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i:
                seq = seq[best_i:] + seq[:best_i]
        all_points.extend(seq)
        if all_points:
            last_pt = all_points[-1]
        reverse_dir = not reverse_dir
    return all_points

# ------------------ WPML/KML 构建 ------------------
def el(tag, text=None, ns="wpml", attrib=None):
    e = ET.Element(f"{{{NS[ns]}}}{tag}", attrib or {})
    if text is not None:
        e.text = str(text)
    return e

def new_kml_doc():
    kml = ET.Element(f"{{{NS['kml']}}}kml")
    doc = ET.SubElement(kml, f"{{{NS['kml']}}}Document")
    return kml, doc

def new_folder(name):
    f = ET.Element(f"{{{NS['kml']}}}Folder")
    nm = ET.SubElement(f, f"{{{NS['kml']}}}name"); nm.text = name
    f.append(el("templateType", "waypoint"))
    f.append(el("templateId", 0))
    cs = el("waylineCoordinateSysParam")
    cs.append(el("coordinateMode", "WGS84"))
    cs.append(el("heightMode", "relativeToStartPoint"))
    f.append(cs)
    return f

def make_mission_config(alt_m, speed_mps):
    mc = el("missionConfig")
    mc.append(el("flyToWaylineMode","safely"))
    mc.append(el("finishAction","goHome"))
    mc.append(el("exitOnRCLost","goContinue"))
    mc.append(el("takeOffSecurityHeight", max(alt_m*0.1, 5.0)))
    mc.append(el("globalTransitionalSpeed", speed_mps))
    return mc

def add_wp(folder, idx, lon, lat, alt_m, speed_mps, heading_deg, gimbal_pitch_deg, with_action=True, file_suffix="LiangchaoDeng_SHZU"):
    pm = ET.SubElement(folder, f"{{{NS['kml']}}}Placemark")
    pt = ET.SubElement(pm, f"{{{NS['kml']}}}Point")
    coord = ET.SubElement(pt, f"{{{NS['kml']}}}coordinates")
    coord.text = f"{lon:.8f},{lat:.8f}"
    pm.append(el("index", idx))
    pm.append(el("useGlobalHeight", 0))
    pm.append(el("ellipsoidHeight", alt_m))
    pm.append(el("height", alt_m))
    pm.append(el("useGlobalSpeed", 1))
    pm.append(el("useGlobalTurnParam", 0))
    wt = el("waypointTurnParam")
    wt.append(el("waypointTurnMode", "toPointAndStopWithDiscontinuityCurvature"))
    wt.append(el("waypointTurnDampingDist", 0.0))
    pm.append(wt)
    pm.append(el("useGlobalHeadingParam", 0))
    wh = el("waypointHeadingParam")
    wh.append(el("waypointHeadingMode", "smoothTransition"))
    wh.append(el("waypointHeadingAngle", float(f"{heading_deg:.1f}")))
    pm.append(wh)
    pm.append(el("gimbalPitchAngle", float(f"{gimbal_pitch_deg:.1f}")))
    if with_action:
        ag = el("actionGroup")
        ag.append(el("actionGroupId", idx))
        ag.append(el("actionGroupStartIndex", idx))
        ag.append(el("actionGroupEndIndex", idx))
        ag.append(el("actionGroupMode","sequence"))
        trg = el("actionTrigger")
        trg.append(el("actionTriggerType","reachPoint"))
        trg.append(el("actionTriggerParam",101))
        ag.append(trg)
        act = el("action")
        act.append(el("actionId",0))
        act.append(el("actionActuatorFunc","takePhoto"))
        ap = el("actionActuatorFuncParam")
        ap.append(el("payloadPositionIndex",0))
        ap.append(el("fileSuffix", file_suffix))
        act.append(ap)
        ag.append(act)
        pm.append(ag)
    return pm

def write_template_kml(path, points, alt_m, speed_mps, gimbal_pitch, device=None):
    kml, doc = new_kml_doc()
    add_device_info_to_doc(doc, device)
    doc.append(make_mission_config(alt_m, speed_mps))
    folder = new_folder("CCO-Template")
    for i,(lon,lat,head) in enumerate(points):
        add_wp(folder, i, lon, lat, alt_m, speed_mps, head, gimbal_pitch, with_action=True)
    doc.append(folder)
    ET.ElementTree(kml).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] template.kml -> {path}")

def write_waylines_wpml(path, points, alt_m, speed_mps, gimbal_pitch, file_suffix="LiangchaoDeng_SHZU", device=None):
    kml, doc = new_kml_doc()
    add_device_info_to_doc(doc, device)
    doc.append(make_mission_config(alt_m, speed_mps))
    folder = new_folder("CCO-Waylines")
    for i,(lon,lat,head) in enumerate(points):
        add_wp(folder, i, lon, lat, alt_m, speed_mps, head, gimbal_pitch, with_action=True, file_suffix=file_suffix)
    doc.append(folder)
    ET.ElementTree(kml).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] waylines.wpml -> {path}")

def write_kmz(kmz_path, template_path, waylines_path):
    build_root = os.path.join(os.path.dirname(kmz_path), "_cco_build")
    wpmz_dir = os.path.join(build_root, "wpmz")
    res_dir  = os.path.join(wpmz_dir, "res")
    os.makedirs(res_dir, exist_ok=True)
    shutil.copyfile(template_path, os.path.join(wpmz_dir, "template.kml"))
    shutil.copyfile(waylines_path, os.path.join(wpmz_dir, "waylines.wpml"))
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wpmz/", "")
        zf.writestr("wpmz/res/", "")
        zf.write(os.path.join(wpmz_dir, "template.kml"), arcname="wpmz/template.kml")
        zf.write(os.path.join(wpmz_dir, "waylines.wpml"), arcname="wpmz/waylines.wpml")
    shutil.rmtree(build_root, ignore_errors=True)
    print(f"[OK] KMZ -> {kmz_path}")

def write_slice_files(out_dir, base_tag, idx1, points, alt_m, speed_mps, gimbal_pitch, file_suffix, device=None):
    os.makedirs(out_dir, exist_ok=True)
    t_path = os.path.join(out_dir, f"{base_tag}_part{idx1:02d}.kml")
    w_path = os.path.join(out_dir, f"{base_tag}_part{idx1:02d}.wpml")
    write_template_kml(t_path, points, alt_m, speed_mps, gimbal_pitch, device=device)
    write_waylines_wpml(w_path, points, alt_m, speed_mps, gimbal_pitch, file_suffix=file_suffix, device=device)
    return t_path, w_path

# ------------------ 预览绘图 ------------------
def show_preview(polygon_lonlat, points_lonlat_head, center_lon, center_lat, title="CCO Preview"):
    if plt is None:
        print("[PREVIEW] matplotlib not installed. Run `pip install matplotlib`. Skipping preview.")
        return
    if not polygon_lonlat or not points_lonlat_head:
        print("[PREVIEW] Empty data. Skipping preview.")
        return
    poly_x = [p[0] for p in polygon_lonlat] + [polygon_lonlat[0][0]]
    poly_y = [p[1] for p in polygon_lonlat] + [polygon_lonlat[0][1]]
    xs = [p[0] for p in points_lonlat_head]
    ys = [p[1] for p in points_lonlat_head]
    plt.figure()
    plt.plot(poly_x, poly_y, linestyle='--', linewidth=1)
    plt.plot(xs, ys, linewidth=1)
    plt.scatter(xs, ys, s=8)
    try:
        centers = globals().get("__preview_centers", None)
        if centers:
            cx = [c[0] for c in centers]; cy = [c[1] for c in centers]
            plt.scatter(cx, cy, s=10, alpha=0.5)
    except Exception:
        pass
    plt.scatter([center_lon], [center_lat], marker='x')
    plt.annotate("start", (xs[0], ys[0]))
    plt.annotate("end", (xs[-1], ys[-1]))
    plt.title(title); plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.axis('equal'); plt.tight_layout(); plt.show()

# ------------------ 保存静态 PNG 预览（无窗口） ------------------
def save_preview_png(polygon_lonlat, points_lonlat_head, centers_lonlat, center_lon, center_lat, out_png):
    if plt is None:
        print(f"[PREVIEW] matplotlib not installed. Cannot save preview: {out_png}")
        return
    # Build static figure (Agg backend works)
    poly_x = [p[0] for p in polygon_lonlat] + [polygon_lonlat[0][0]]
    poly_y = [p[1] for p in polygon_lonlat] + [polygon_lonlat[0][1]]
    xs = [p[0] for p in points_lonlat_head]
    ys = [p[1] for p in points_lonlat_head]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poly_x, poly_y, linestyle='--', linewidth=1)
    if centers_lonlat:
        cx = [c[0] for c in centers_lonlat]; cy = [c[1] for c in centers_lonlat]
        ax.scatter(cx, cy, s=10, alpha=0.5)
    if xs and ys:
        ax.plot(xs, ys, linewidth=1)
        ax.scatter(xs, ys, s=8)
        ax.annotate("start", (xs[0], ys[0]))
        ax.annotate("end", (xs[-1], ys[-1]))
    ax.scatter([center_lon], [center_lat], marker='x')
    ax.set_title("CCO Preview (static)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.axis('equal'); fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[PREVIEW] saved PNG -> {out_png}")

# ------------------ 交互式 GUI 预览 ------------------
def show_preview_gui(poly, center_lon, center_lat, init_params):
    if plt is None or Slider is None or Button is None:
        print("[PREVIEW GUI] GUI backend not available. Install Tk>=8.6 or PyQt5/PySide6 + matplotlib.")
        return init_params
    # unpack
    circle_radius = init_params["circle_radius"]
    per_ring = init_params["per_ring"]
    overlap = init_params["circle_overlap"]
    center_step = init_params["center_step"]
    cover_padding = init_params["cover_padding"]
    grid_bearing = init_params["grid_bearing"]
    clip_inside = init_params["clip_inside"]
    prune_outside = init_params.get("prune_outside", 1)
    max_points_gui = int(init_params.get("max_points", 300))

    # mutable state for polygon and center (allow replacing via file picker)
    poly_ref = {"poly": poly, "lon_c": center_lon, "lat_c": center_lat}

    fig, ax = plt.subplots(figsize=(11, 7))
    # Leave space on the right for a wider controls panel
    plt.subplots_adjust(left=0.08, right=0.70, top=0.96, bottom=0.08)
    # polygon
    poly_x = [p[0] for p in poly_ref["poly"]] + [poly_ref["poly"][0][0]]
    poly_y = [p[1] for p in poly_ref["poly"]] + [poly_ref["poly"][0][1]]
    poly_line, = ax.plot(poly_x, poly_y, linestyle='--', linewidth=1)
    centers_scatter = ax.scatter([], [], s=10, alpha=0.5)
    path_line, = ax.plot([], [], linewidth=1)
    pts_scatter = ax.scatter([], [], s=8)
    center_scatter = ax.scatter([poly_ref["lon_c"]], [poly_ref["lat_c"]], marker='x')
    ax.set_title("CCO Preview (Interactive)")
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.axis('equal')

    axcolor = 'lightgoldenrodyellow'
    # Right-side control panel positions (wider panel to prevent label overflow)
    _px, _pw = 0.71, 0.27
    # Sliders (top to bottom)
    ax_max  = plt.axes([_px, 0.90, _pw, 0.035], facecolor=axcolor)
    ax_r    = plt.axes([_px, 0.86, _pw, 0.03],  facecolor=axcolor)
    ax_pr   = plt.axes([_px, 0.82, _pw, 0.03],  facecolor=axcolor)
    ax_ov   = plt.axes([_px, 0.78, _pw, 0.03],  facecolor=axcolor)
    ax_step = plt.axes([_px, 0.74, _pw, 0.03],  facecolor=axcolor)
    ax_pad  = plt.axes([_px, 0.70, _pw, 0.03],  facecolor=axcolor)
    ax_bear = plt.axes([_px, 0.66, _pw, 0.03],  facecolor=axcolor)

    s_r   = Slider(ax_r,   'Radius',      1.0, 200.0,   valinit=circle_radius, valstep=0.5)
    s_pr  = Slider(ax_pr,  'Pts/circ',    6,   200,     valinit=per_ring,      valstep=1)
    s_ov  = Slider(ax_ov,  'Overlap',     0.0, 0.9,     valinit=overlap,       valstep=0.01)
    s_step= Slider(ax_step,'Center step', 0.0, 300.0,   valinit=center_step,   valstep=1.0)
    s_pad = Slider(ax_pad, 'Padding',     0.0, 100.0,   valinit=cover_padding, valstep=1.0)
    s_bear= Slider(ax_bear,'Bearing (°)', -180.0, 180.0,valinit=grid_bearing,  valstep=1.0)
    s_max = Slider(ax_max, 'Max points',  50,  2000,    valinit=float(max_points_gui), valstep=1)
    s_max.on_changed(lambda v: None)  # 不触发重算，只在 Apply 时生效

    # ---- Keyboard control for sliders ----
    sliders = [
        (s_r,   'Radius(m)'),
        (s_pr,  'Pts/Circle'),
        (s_ov,  'Overlap'),
        (s_step,'Center step(m)'),
        (s_pad, 'Padding(m)'),
        (s_bear,'Grid bearing(deg)'),
        (s_max, 'Max points/sub-mission'),
    ]
    active_idx = {"i": 0}

    def _update_active_label():
        # add a leading "* " to the active slider label for a clear cue
        for idx, (sl, name) in enumerate(sliders):
            lbl = name
            if idx == active_idx["i"]:
                lbl = f"* {name}"
            # update text if changed
            if sl.label.get_text() != lbl:
                sl.label.set_text(lbl)
        fig.canvas.draw_idle()

    def _step_of(sl):
        # derive step from valstep if available; else fallback by slider type
        st = getattr(sl, 'valstep', None)
        if st is None or st == 0:
            # fallback heuristics by label
            nm = sl.label.get_text().lstrip('* ').lower()
            if 'radius' in nm:
                return 0.5
            if 'pts/circle' in nm or 'max points' in nm:
                return 1
            if 'overlap' in nm:
                return 0.01
            if 'step' in nm or 'padding' in nm or 'bearing' in nm:
                return 1.0
            return 1.0
        try:
            return float(st)
        except Exception:
            return 1.0

    def _clamp(v, vmin, vmax):
        return max(vmin, min(vmax, v))

    def _set_slider(sl, newv):
        # use set_val to trigger on_changed
        sl.set_val(newv)

    def on_key(event):
        if event.key is None:
            return
        key = event.key.lower()
        # Switch active slider with Tab / Shift+Tab / Up / Down
        if key in ('tab', 'down'):
            active_idx["i"] = (active_idx["i"] + 1) % len(sliders)
            _update_active_label()
            return
        if key in ('shift+tab', 'up'):
            active_idx["i"] = (active_idx["i"] - 1) % len(sliders)
            _update_active_label()
            return
        # Fine adjust with Left/Right; Shift for x10
        if key.startswith('left') or key.startswith('right'):
            sl, _ = sliders[active_idx["i"]]
            step = _step_of(sl)
            if 'shift+' in key:
                step *= 10.0
            cur = float(sl.val)
            if key.startswith('left'):
                newv = cur - step
            else:
                newv = cur + step
            newv = _clamp(newv, sl.valmin, sl.valmax)
            _set_slider(sl, newv)
            return

    # initial label mark and key binding
    _update_active_label()
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Buttons on right panel
    ax_clip  = plt.axes([_px, 0.33, 0.12, 0.05]); btn_clip  = Button(ax_clip,  f"Clip: {'ON' if clip_inside else 'OFF'}")
    ax_prune = plt.axes([_px + 0.14, 0.33, 0.12, 0.05]); btn_prune = Button(ax_prune, f"Prune: {'ON' if prune_outside else 'OFF'}")
    ax_open  = plt.axes([_px, 0.26, _pw, 0.05]); btn_open  = Button(ax_open,  "Open KML…")
    ax_apply = plt.axes([_px, 0.18, _pw, 0.06]); btn_apply = Button(ax_apply, "Apply & Generate")

    # Prefill device defaults if not provided
    def _def(v, dv):
        return str(v) if (v is not None and str(v).strip() != "") else str(dv)
    init_de  = _def(init_params.get("drone_enum"),   99)
    init_dse = _def(init_params.get("drone_sub"),     1)
    init_pe  = _def(init_params.get("payload_enum"), 89)
    init_pse = _def(init_params.get("payload_sub"),   0)
    init_pp  = _def(init_params.get("payload_pos"),   0)
    # TextBoxes on right panel (stacked)
    ax_de  = plt.axes([_px, 0.60, _pw, 0.045]); tb_de  = TextBox(ax_de,  "DroneEnum",   initial=init_de)
    ax_dse = plt.axes([_px, 0.55, _pw, 0.045]); tb_dse = TextBox(ax_dse, "DroneSub",    initial=init_dse)
    ax_pe  = plt.axes([_px, 0.50, _pw, 0.045]); tb_pe  = TextBox(ax_pe,  "PayloadEnum", initial=init_pe)
    ax_pse = plt.axes([_px, 0.45, _pw, 0.045]); tb_pse = TextBox(ax_pse, "PayloadSub",  initial=init_pse)
    ax_pp  = plt.axes([_px, 0.40, _pw, 0.045]); tb_pp  = TextBox(ax_pp,  "PayloadPos",  initial=init_pp)

    def recalc_and_draw(_=None):
        R = s_r.val
        PR = max(3, int(s_pr.val))
        OV = s_ov.val
        STEP = s_step.val if s_step.val > 0 else max(2.0*R*(1.0-OV), 1.0)
        PAD = s_pad.val
        BEAR = s_bear.val
        centers = grid_circle_centers(poly_ref["poly"], poly_ref["lon_c"], poly_ref["lat_c"], STEP, PAD, BEAR)
        if not centers:
            centers_scatter.set_offsets([])
            path_line.set_data([], [])
            pts_scatter.set_offsets([])
            fig.canvas.draw_idle()
            return
        # Prune if Clip is OFF and Prune is ON
        if btn_clip.label.get_text().endswith('OFF') and btn_prune.label.get_text().endswith('ON'):
            centers = prune_centers_outside(poly_ref["poly"], centers, R, PR, 0.0)
        xs_c = [c[0] for c in centers]; ys_c = [c[1] for c in centers]
        centers_scatter.set_offsets(list(zip(xs_c, ys_c)))
        pts = generate_cover_cco_points(
            centers, per_circle=PR, circle_radius_m=R, start_bearing_deg=0.0,
            clip_poly=poly_ref["poly"] if btn_clip.label.get_text().endswith('ON') else None
        )
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        path_line.set_data(xs, ys)
        pts_scatter.set_offsets(list(zip(xs, ys)))
        ax.relim(); ax.autoscale_view()
        fig.canvas.draw_idle()

    def on_clip(event):
        # toggle
        txt = btn_clip.label.get_text()
        btn_clip.label.set_text("Clip: OFF" if txt.endswith("ON") else "Clip: ON")
        recalc_and_draw()

    def on_prune(event):
        txt = btn_prune.label.get_text()
        btn_prune.label.set_text("Prune: OFF" if txt.endswith("ON") else "Prune: ON")
        recalc_and_draw()

    def on_apply(event):
        init_params["max_points"] = int(s_max.val)
        init_params["circle_radius"] = float(s_r.val)
        init_params["per_ring"] = int(s_pr.val)
        init_params["circle_overlap"] = float(s_ov.val)
        init_params["center_step"] = float(s_step.val)
        init_params["cover_padding"] = float(s_pad.val)
        init_params["grid_bearing"] = float(s_bear.val)
        init_params["clip_inside"] = 1 if btn_clip.label.get_text().endswith("ON") else 0
        init_params["prune_outside"] = 1 if btn_prune.label.get_text().endswith("ON") else 0
        def _to_int_or_none(s):
            try:
                v = str(s).strip()
                return int(v) if v != "" else None
            except Exception:
                return None
        init_params["drone_enum"] = _to_int_or_none(tb_de.text)
        init_params["drone_sub"] = _to_int_or_none(tb_dse.text)
        init_params["payload_enum"] = _to_int_or_none(tb_pe.text)
        init_params["payload_sub"] = _to_int_or_none(tb_pse.text)
        init_params["payload_pos"] = _to_int_or_none(tb_pp.text)
        plt.close(fig)

    def on_open(event):
        kml_path = pick_open_file_kml()
        if not kml_path:
            return
        try:
            new_poly = parse_polygon_kml(kml_path)
            lon_c, lat_c = polygon_centroid(new_poly)
            poly_ref["poly"] = new_poly
            poly_ref["lon_c"] = lon_c
            poly_ref["lat_c"] = lat_c
            px = [p[0] for p in new_poly] + [new_poly[0][0]]
            py = [p[1] for p in new_poly] + [new_poly[0][1]]
            poly_line.set_data(px, py)
            center_scatter.set_offsets([[lon_c, lat_c]])
            recalc_and_draw()
        except Exception as e:
            print(f"[GUI] Failed to load KML: {e}")

    s_r.on_changed(recalc_and_draw)
    s_pr.on_changed(recalc_and_draw)
    s_ov.on_changed(recalc_and_draw)
    s_step.on_changed(recalc_and_draw)
    s_pad.on_changed(recalc_and_draw)
    s_bear.on_changed(recalc_and_draw)
    btn_clip.on_clicked(on_clip)
    btn_prune.on_clicked(on_prune)
    btn_apply.on_clicked(on_apply)
    btn_open.on_clicked(on_open)

    recalc_and_draw()
    plt.show()
    return init_params

def standalone_cli_fallback():
    def _ask_opt_int(prompt):
        try:
            s = input(f"{prompt} (empty=skip): ").strip()
            return int(s) if s != "" else None
        except Exception:
            return None
    print("[STANDALONE] Tk < 8.6 detected. Switching to CLI fallback (text prompts).")
    # Prompt for essential paths
    try:
        kml_path = input("Enter path to Polygon KML (e.g., ./2025sj.kml): ").strip()
    except EOFError:
        kml_path = ""
    if not kml_path:
        print("[STANDALONE] No KML path provided. Exit.")
        return
    try:
        out_dir = input("Enter output directory (default ./output): ").strip()
    except EOFError:
        out_dir = ""
    if not out_dir:
        out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)

    # Optional param prompts with defaults
    def _ask_float(prompt, default):
        try:
            s = input(f"{prompt} [{default}]: ").strip()
            return float(s) if s else float(default)
        except Exception:
            return float(default)
    def _ask_int(prompt, default):
        try:
            s = input(f"{prompt} [{default}]: ").strip()
            return int(s) if s else int(default)
        except Exception:
            return int(default)
    def _ask_bool01(prompt, default):
        try:
            s = input(f"{prompt} (1/0) [{default}]: ").strip()
            return 1 if (s=="1" or (not s and default==1)) else 0
        except Exception:
            return default

    circle_radius = _ask_float("Circle radius (m)", 30.0)
    per_ring = _ask_int("Points per circle", 60)
    circle_overlap = _ask_float("Circle overlap (0~0.9)", 0.3)
    center_step = _ask_float("Center step (m, 0=auto)", 0.0)
    cover_padding = _ask_float("Padding (m)", 10.0)
    grid_bearing = _ask_float("Grid bearing (deg)", 0.0)
    clip_inside = _ask_bool01("Clip inside polygon", 1)
    alt = _ask_float("Altitude (m)", 50.0)
    speed = _ask_float("Speed (m/s)", 6.0)
    gimbal = _ask_float("Gimbal pitch (deg)", -45.0)
    max_points = _ask_int("Max points per sub-mission", 300)

    drone_enum = _ask_opt_int("Drone enum")
    drone_sub = _ask_opt_int("Drone sub-enum")
    payload_enum = _ask_opt_int("Payload enum")
    payload_sub = _ask_opt_int("Payload sub-enum")
    payload_pos = _ask_opt_int("Payload position index")
    device = {
        "drone_enum": drone_enum,
        "drone_sub": drone_sub,
        "payload_enum": payload_enum,
        "payload_sub": payload_sub,
        "payload_pos": payload_pos,
    }

    # Parse & generate
    poly = parse_polygon_kml(kml_path)
    lon_c, lat_c = polygon_centroid(poly)
    step_m = center_step if center_step > 0 else max(2.0 * circle_radius * (1.0 - circle_overlap), 1.0)
    centers = grid_circle_centers(poly, lon_c, lat_c, step_m=step_m, padding_m=cover_padding, bearing_deg=grid_bearing)
    pts = generate_cover_cco_points(
        centers,
        per_circle=per_ring,
        circle_radius_m=circle_radius,
        start_bearing_deg=0.0,
        clip_poly=poly if clip_inside else None
    )

    # Write outputs
    tpl = os.path.join(out_dir, "template.kml")
    wpl = os.path.join(out_dir, "waylines.wpml")
    write_template_kml(tpl, pts, alt, speed, gimbal, device=device)
    write_waylines_wpml(wpl, pts, alt, speed, gimbal, file_suffix="LiangchaoDeng_SHZU", device=device)
    write_kmz(os.path.join(out_dir, "cco_full.kmz"), tpl, wpl)

    # Split if needed
    total = len(pts)
    if max_points and max_points < total:
        cuts = chunk_slices(total, max_points)
        parts_dir = os.path.join(out_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        for pi, (s, e) in enumerate(cuts, start=1):
            part_pts = pts[s:e]
            t_path, w_path = write_slice_files(parts_dir, "template", pi, part_pts, alt, speed, gimbal, "LiangchaoDeng_SHZU", device=device)
            build_kmz_for_part(parts_dir, f"part{pi:02d}", t_path, w_path)
        print(f"[SPLIT] total {total} waypoints; split by {max_points} -> {len(cuts)} parts; output: {parts_dir}")
    else:
        print(f"[INFO] total waypoints {total} ≤ max_points={max_points}; no split needed.")

    # Save static preview PNG (no GUI required)
    try:
        out_png = os.path.join(out_dir, "preview.png")
        save_preview_png(poly, pts, centers, lon_c, lat_c, out_png)
    except Exception as e:
        print(f"[PREVIEW] failed to save PNG: {e}")

    print(f"[DONE] Output directory: {out_dir}")

# ------------------ Standalone（无参数直接运行） ------------------
def standalone_flow():
    """
    Run without CLI args:
    - Pick polygon KML
    - (Optional) pick output folder
    - Open interactive preview GUI
    - On Apply & Generate: write outputs
    """
    # If neither Tk(>=8.6) nor Qt is available, fallback to CLI
    if (not TK_OK) and (not USE_QT):
        return standalone_cli_fallback()

    # file pickers (Tk or Qt)
    kml_path = pick_open_file_kml()
    if not kml_path:
        print("[STANDALONE] No KML selected. Exit.")
        return
    out_dir = pick_select_folder()
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(kml_path), "output")
    os.makedirs(out_dir, exist_ok=True)

    # parse polygon & center
    poly = parse_polygon_kml(kml_path)
    lon_c, lat_c = polygon_centroid(poly)

    # default params for first preview
    params = {
        "circle_radius": 30.0,
        "per_ring": 60,
        "circle_overlap": 0.3,
        "center_step": 0.0,      # 0 => auto from radius & overlap
        "cover_padding": 10.0,
        "grid_bearing": 0.0,
        "clip_inside": 1,
        "prune_outside": 1,
        "max_points": 300,
        # Defaults for Matrice 4T + payload
        "drone_enum": 99,
        "drone_sub": 1,
        "payload_enum": 89,
        "payload_sub": 0,
        "payload_pos": 0,
    }
    params = show_preview_gui(poly, lon_c, lat_c, params)

    # compute point sequence with chosen params
    step_m = params["center_step"] if params["center_step"] > 0 else max(2.0 * params["circle_radius"] * (1.0 - params["circle_overlap"]), 1.0)
    centers = grid_circle_centers(poly, lon_c, lat_c, step_m=step_m, padding_m=params["cover_padding"], bearing_deg=params["grid_bearing"])
    # Prune centers if needed
    if (not params.get("clip_inside", 1)) and params.get("prune_outside", 1):
        centers = prune_centers_outside(poly, centers, params["circle_radius"], params["per_ring"], 0.0)
    globals()["__preview_centers"] = centers
    pts = generate_cover_cco_points(
        centers,
        per_circle=params["per_ring"],
        circle_radius_m=params["circle_radius"],
        start_bearing_deg=0.0,
        clip_poly=poly if params["clip_inside"] else None
    )

    # write outputs (use sensible defaults)
    alt = 50.0; speed = 6.0; gimbal = -45.0; file_suffix = "LiangchaoDeng_SHZU"
    tpl = os.path.join(out_dir, "template.kml")
    wpl = os.path.join(out_dir, "waylines.wpml")
    device = {
        "drone_enum": params.get("drone_enum"),
        "drone_sub": params.get("drone_sub"),
        "payload_enum": params.get("payload_enum"),
        "payload_sub": params.get("payload_sub"),
        "payload_pos": params.get("payload_pos"),
    }
    write_template_kml(tpl, pts, alt, speed, gimbal, device=device)
    write_waylines_wpml(wpl, pts, alt, speed, gimbal, file_suffix=file_suffix, device=device)
    write_kmz(os.path.join(out_dir, "cco_full.kmz"), tpl, wpl)

    # split if needed (default 300)
    # split if needed (GUI-chosen max points)
    max_points = int(params.get("max_points", 300))
    total = len(pts)
    if max_points and max_points < total:
        cuts = chunk_slices(total, max_points)
        parts_dir = os.path.join(out_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        for pi, (s, e) in enumerate(cuts, start=1):
            part_pts = pts[s:e]
            t_path, w_path = write_slice_files(parts_dir, "template", pi, part_pts, alt, speed, gimbal, file_suffix, device=device)
            build_kmz_for_part(parts_dir, f"part{pi:02d}", t_path, w_path)
        print(f"[SPLIT] total {total} waypoints; split by {max_points} -> {len(cuts)} parts; output: {parts_dir}")
    else:
        print(f"[INFO] total waypoints {total} ≤ max_points={max_points}; no split needed.")

    print(f"[DONE] Output directory: {out_dir}")

# ------------------ CLI ------------------
def main():
    # 无参数：进入 Standalone GUI
    if len(sys.argv) == 1:
        standalone_flow()
        return

    ap = argparse.ArgumentParser(description="Generate cover-style CCO routes from a KML Polygon (template.kml + waylines.wpml + KMZ)")
    ap.add_argument("--polygon", required=True, help="KML file containing a Polygon")
    ap.add_argument("--center_mode", choices=["centroid","bbox_center"], default="centroid", help="Center point: centroid or bounding-box center")
    ap.add_argument("--per_ring", type=int, default=60, help="Points per circle (cover mode)")
    ap.add_argument("--start_bearing", type=float, default=0.0, help="Start bearing in degrees (0=N, clockwise)")
    ap.add_argument("--alt", type=float, default=50.0, help="Altitude (m)")
    ap.add_argument("--speed", type=float, default=6.0, help="Speed (m/s)")
    ap.add_argument("--gimbal", type=float, default=-45.0, help="Gimbal pitch (deg, negative=down)")
    ap.add_argument("--file_suffix", default="LiangchaoDeng_SHZU", help="Photo filename suffix")
    # Output/split/pack/preview
    ap.add_argument("--kmz", type=int, default=1, help="Whether to generate KMZ (1=yes)")
    ap.add_argument("--out_dir", default="output", help="Output directory (default ./output)")
    ap.add_argument("--max_points", type=int, default=300, help="Max waypoints per sub-mission (default 300)")
    ap.add_argument("--pack_full_kmz", type=int, default=1, help="Pack full KMZ (1=yes)")
    ap.add_argument("--pack_parts_kmz", type=int, default=1, help="Pack split KMZ parts (1=yes)")
    ap.add_argument("--preview", type=int, default=0, help="Show static preview window (1=yes)")
    # Cover mode params
    ap.add_argument("--circle_radius", type=float, default=30.0, help="Circle radius (m) for cover mode")
    ap.add_argument("--center_step", type=float, default=0.0, help="Center spacing (m); 0 = auto from radius & overlap")
    ap.add_argument("--circle_overlap", type=float, default=0.3, help="Linear overlap between adjacent circles (0~0.9)")
    ap.add_argument("--cover_padding", type=float, default=10.0, help="Padding (m) to expand bbox to avoid edge gaps")
    # New params
    ap.add_argument("--preview_gui", type=int, default=0, help="Open interactive preview with sliders (1=yes)")
    ap.add_argument("--grid_bearing", type=float, default=0.0, help="Grid rotation angle in degrees (clockwise)")
    ap.add_argument("--clip_inside", type=int, default=1, help="Keep only waypoints inside polygon (1=yes, 0=no)")
    ap.add_argument("--prune_outside", type=int, default=1, help="When clip=0, drop circles with no waypoint inside polygon (1=yes)")
    ap.add_argument("--drone_enum", type=int, default=None, help="DJI droneEnumValue (integer)")
    ap.add_argument("--drone_sub", type=int, default=None, help="DJI droneSubEnumValue (integer)")
    ap.add_argument("--payload_enum", type=int, default=None, help="DJI payloadEnumValue (integer)")
    ap.add_argument("--payload_sub", type=int, default=None, help="DJI payloadSubEnumValue (integer)")
    ap.add_argument("--payload_pos", type=int, default=None, help="DJI payloadPositionIndex (integer)")
    args = ap.parse_args()

    # 读取多边形并求中心
    poly = parse_polygon_kml(args.polygon)
    if args.center_mode == "bbox_center":
        lons = [p[0] for p in poly]; lats = [p[1] for p in poly]
        lon_c = (min(lons)+max(lons))/2.0; lat_c = (min(lats)+max(lats))/2.0
    else:
        lon_c, lat_c = polygon_centroid(poly)

    # 交互式预览（可选）：允许先调参后生成
    if args.preview_gui:
        params = {
            "circle_radius": args.circle_radius,
            "per_ring": args.per_ring,
            "circle_overlap": args.circle_overlap,
            "center_step": args.center_step,
            "cover_padding": args.cover_padding,
            "grid_bearing": args.grid_bearing,
            "clip_inside": args.clip_inside,
            "max_points": args.max_points,
            "drone_enum": args.drone_enum,
            "drone_sub": args.drone_sub,
            "payload_enum": args.payload_enum,
            "payload_sub": args.payload_sub,
            "payload_pos": args.payload_pos,
        }
        params = show_preview_gui(poly, lon_c, lat_c, params)
        # 覆盖 CLI 值
        args.circle_radius = params["circle_radius"]
        args.per_ring = params["per_ring"]
        args.circle_overlap = params["circle_overlap"]
        args.center_step = params["center_step"]
        args.cover_padding = params["cover_padding"]
        args.grid_bearing = params["grid_bearing"]
        args.clip_inside = params["clip_inside"]
        args.max_points = params.get("max_points", args.max_points)
        args.per_ring = max(3, int(args.per_ring))
        args.drone_enum = params.get("drone_enum")
        args.drone_sub = params.get("drone_sub")
        args.payload_enum = params.get("payload_enum")
        args.payload_sub = params.get("payload_sub")
        args.payload_pos = params.get("payload_pos")

    # enforce per_ring minimum
    args.per_ring = max(3, int(args.per_ring))

    # 生成点序列（覆盖式）
    step_m = args.center_step if args.center_step > 0 else max(2.0 * args.circle_radius * (1.0 - args.circle_overlap), 1.0)
    centers = grid_circle_centers(poly, lon_c, lat_c, step_m=step_m, padding_m=args.cover_padding, bearing_deg=args.grid_bearing)
    # Prune centers if needed
    if (not args.clip_inside) and args.prune_outside:
        centers = prune_centers_outside(poly, centers, args.circle_radius, args.per_ring, args.start_bearing)
    globals()["__preview_centers"] = centers  # 给预览画圆心
    pts = generate_cover_cco_points(
        centers,
        per_circle=args.per_ring,
        circle_radius_m=args.circle_radius,
        start_bearing_deg=args.start_bearing,
        clip_poly=poly if args.clip_inside else None
    )

    # 预览
    if args.preview:
        show_preview(poly, pts, lon_c, lat_c, title=f"CCO Preview (cover, rotate {args.grid_bearing:.1f}°, Clip={'ON' if args.clip_inside else 'OFF'})")

    # 输出目录
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    # 写全量文件
    device = {
        "drone_enum": args.drone_enum,
        "drone_sub": args.drone_sub,
        "payload_enum": args.payload_enum,
        "payload_sub": args.payload_sub,
        "payload_pos": args.payload_pos,
    }
    tpl = os.path.join(out_root, "template.kml")
    wpl = os.path.join(out_root, "waylines.wpml")
    write_template_kml(tpl, pts, args.alt, args.speed, args.gimbal, device=device)
    write_waylines_wpml(wpl, pts, args.alt, args.speed, args.gimbal, file_suffix=args.file_suffix, device=device)

    # 打完整 KMZ（可选）
    if args.kmz and args.pack_full_kmz:
        write_kmz(os.path.join(out_root, "cco_full.kmz"), tpl, wpl)

    # 分片与打包
    total = len(pts)
    if args.max_points and args.max_points < total:
        cuts = chunk_slices(total, args.max_points)
        parts_dir = os.path.join(out_root, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        for pi, (s, e) in enumerate(cuts, start=1):
            part_pts = pts[s:e]
            t_path, w_path = write_slice_files(parts_dir, "template", pi, part_pts, args.alt, args.speed, args.gimbal, args.file_suffix, device=device)
            if args.pack_parts_kmz:
                build_kmz_for_part(parts_dir, f"part{pi:02d}", t_path, w_path)
        print(f"[SPLIT] total {total} waypoints; split by {args.max_points} -> {len(cuts)} parts; output: {parts_dir}")
    else:
        print(f"[INFO] total waypoints {total} ≤ max_points={args.max_points}; no split needed.")

if __name__ == "__main__":
    main()