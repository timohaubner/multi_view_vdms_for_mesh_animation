#!/usr/bin/env python3
"""
verify_camera.py

Rendert ein Mesh mit pyrender, projiziert dann die 3D-Vertices
mit den Kalibrationsparametern auf das Bild und überlagert beides.

Wenn die Kalibration stimmt, liegen alle projizierten Punkte
exakt auf der gerenderten Silhouette.
"""

from pathlib import Path
import argparse
import numpy as np
import trimesh
import pyrender
import cv2


# ─── Funktionen aus dem Kalibrations-Skript (identisch kopiert) ───


def get_intrinsics(width, height, yfov):
    fy = height / (2.0 * np.tan(yfov / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def get_camera_position(center, size, azimuth_deg):
    radius = 1.45 * size
    height_offset = 0.25 * size
    azimuth_rad = np.deg2rad(azimuth_deg)
    x = np.sin(azimuth_rad) * radius
    z = np.cos(azimuth_rad) * radius
    y = height_offset
    return center + np.array([x, y, z], dtype=np.float64)


def look_at(camera_pos, target, up=np.array([0.0, 1.0, 0.0])):
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = right
    T[:3, 1] = true_up
    T[:3, 2] = -forward
    T[:3, 3] = camera_pos
    return T


def get_camera_calibration(first_mesh_path, width, height, azimuth_deg, yfov=np.pi / 4.0):
    mesh = trimesh.load(first_mesh_path, process=False)
    center = mesh.bounds.mean(axis=0).astype(np.float64)
    size = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    camera_pos = get_camera_position(center, size, azimuth_deg)
    target = center + np.array([0.0, 0.05 * size, 0.0], dtype=np.float64)

    T_wc_gl = look_at(camera_pos, target)
    T_cw_gl = np.linalg.inv(T_wc_gl)

    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    T_cw_cv = gl_to_cv @ T_cw_gl

    R = T_cw_cv[:3, :3]
    t = T_cw_cv[:3, 3:]
    K = get_intrinsics(width, height, yfov)

    return {
        "K": K,
        "R": R,
        "t": t,
        "T_wc_gl": T_wc_gl,
        "yfov": yfov,
    }


# ─── Rendering (identisch zum Rendering-Skript) ───


def render_mesh(mesh_path, width, height, azimuth_deg, yfov=np.pi / 4.0):
    """Rendert das Mesh und gibt (color_image, depth_image) zurück."""

    tri = trimesh.load(mesh_path, process=False)
    center = tri.bounds.mean(axis=0)
    size = np.linalg.norm(tri.bounds[1] - tri.bounds[0])

    camera_pos = get_camera_position(center, size, azimuth_deg)
    target = center + np.array([0.0, 0.05 * size, 0.0])
    camera_pose = look_at(camera_pos, target)

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[0.6, 0.6, 0.6],
    )

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.35, 0.35, 0.35, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.75,
    )

    render_mesh = pyrender.Mesh.from_trimesh(tri, material=material, smooth=True)
    scene.add(render_mesh)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()

    return color, depth


# ─── Projektion ───


def project_vertices(vertices, K, R, t):
    """
    Projiziert 3D-Punkte (N×3) auf 2D-Pixelkoordinaten.

    X_cam = R @ X_world + t
    u = fx * X_cam / Z_cam + cx
    v = fy * Y_cam / Z_cam + cy
    """
    pts_cam = (R @ vertices.T + t).T          # (N, 3)

    # Nur Punkte vor der Kamera (Z > 0)
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]

    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]

    return np.stack([u, v], axis=1), valid


# ─── Tests ───


def test_projection_on_image(mesh_path, width, height, azimuth_deg, yfov, out_path):
    """
    TEST 1: Projizierte Vertices auf gerendertes Bild zeichnen.
    Visuell prüfen ob die Punkte auf der Mesh-Oberfläche liegen.
    """
    print("=" * 60)
    print("TEST 1: Projektion auf gerendertes Bild")
    print("=" * 60)

    # Rendern
    color, depth = render_mesh(mesh_path, width, height, azimuth_deg, yfov)

    # Kalibration holen
    cal = get_camera_calibration(mesh_path, width, height, azimuth_deg, yfov)

    # Vertices laden
    tri = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(tri.vertices, dtype=np.float64)

    # Projizieren
    pts_2d, valid = project_vertices(vertices, cal["K"], cal["R"], cal["t"])

    # Auf Bild zeichnen
    img = color.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for pt in pts_2d:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)   # rot

    cv2.imwrite(str(out_path), img)
    print(f"  Overlay gespeichert: {out_path}")
    print(f"  Vertices total: {len(vertices)}")
    print(f"  Vertices vor Kamera: {valid.sum()}")
    print(f"  Vertices im Bild: {((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < width) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < height)).sum()}")


def test_depth_consistency(mesh_path, width, height, azimuth_deg, yfov):
    """
    Verbesserter Depth-Test: Filtert selbstverdeckte Vertices heraus.

    Ein Vertex gilt als "sichtbar", wenn sein projiziertes Z
    nahe am Depth-Buffer liegt (nicht deutlich dahinter).
    """
    print()
    print("=" * 60)
    print("TEST 2b: Tiefenkonsistenz (nur sichtbare Vertices)")
    print("=" * 60)

    color, depth = render_mesh(mesh_path, width, height, azimuth_deg, yfov)
    cal = get_camera_calibration(mesh_path, width, height, azimuth_deg, yfov)

    tri = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(tri.vertices, dtype=np.float64)

    pts_cam = (cal["R"] @ vertices.T + cal["t"]).T
    K = cal["K"]

    errors_visible = []
    errors_all = []
    n_occluded = 0
    n_compared = 0

    for i in range(len(pts_cam)):
        z_proj = pts_cam[i, 2]
        if z_proj <= 0:
            continue

        u = K[0, 0] * pts_cam[i, 0] / z_proj + K[0, 2]
        v = K[1, 1] * pts_cam[i, 1] / z_proj + K[1, 2]

        px, py = int(round(u)), int(round(v))
        if 0 <= px < width and 0 <= py < height:
            z_depth = depth[py, px]
            if z_depth <= 0:
                continue

            n_compared += 1
            error = abs(z_proj - z_depth)
            errors_all.append(error)

            # Vertex ist sichtbar wenn sein Z ≈ Depth-Buffer
            # (nicht deutlich hinter der sichtbaren Oberfläche)
            if z_proj <= z_depth + 0.02:  # 2cm Toleranz
                errors_visible.append(error)
            else:
                n_occluded += 1

    errors_visible = np.array(errors_visible)
    errors_all = np.array(errors_all)

    print(f"  Verglichene Punkte:    {n_compared}")
    print(f"  Selbstverdeckt:        {n_occluded}")
    print(f"  Sichtbare Vertices:    {len(errors_visible)}")
    print()

    if len(errors_visible) > 0:
        print(f"  Nur sichtbare Vertices:")
        print(f"    Z-Fehler (Median):   {np.median(errors_visible):.6f}")
        print(f"    Z-Fehler (Mean):     {np.mean(errors_visible):.6f}")
        print(f"    Z-Fehler (Max):      {np.max(errors_visible):.6f}")
        print(f"    Z-Fehler (95%):      {np.percentile(errors_visible, 95):.6f}")

        if np.median(errors_visible) < 0.005:
            print("  ✅ BESTANDEN – Tiefen sichtbarer Vertices stimmen überein")
        else:
            print("  ❌ FEHLER – Tiefen weichen ab!")

    print()
    print(f"  Alle Vertices (inkl. verdeckte):")
    print(f"    Z-Fehler (Median):   {np.median(errors_all):.6f}")
    print(f"    Z-Fehler (Mean):     {np.mean(errors_all):.6f}")
    print(f"    Anteil verdeckt:     {n_occluded}/{n_compared}"
          f" ({100 * n_occluded / n_compared:.1f}%)")


def test_reprojection_identity(mesh_path, width, height, azimuth_deg, yfov):
    """
    TEST 3: Rein mathematischer Test.

    project(unproject(pixel)) == pixel  (für Punkte auf dem Mesh)

    Verwendet cv2.projectPoints mit den Kalibrationsparametern.
    """
    print()
    print("=" * 60)
    print("TEST 3: cv2.projectPoints vs. manuelle Projektion")
    print("=" * 60)

    cal = get_camera_calibration(mesh_path, width, height, azimuth_deg, yfov)

    tri = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(tri.vertices, dtype=np.float64)

    # Manuelle Projektion
    pts_manual, valid = project_vertices(vertices, cal["K"], cal["R"], cal["t"])

    # cv2.projectPoints
    rvec, _ = cv2.Rodrigues(cal["R"])
    dist = np.zeros(5)

    pts_cv2, _ = cv2.projectPoints(
        vertices[valid],
        rvec,
        cal["t"],
        cal["K"],
        dist,
    )
    pts_cv2 = pts_cv2.reshape(-1, 2)

    diff = np.linalg.norm(pts_manual - pts_cv2, axis=1)

    print(f"  Punkte verglichen: {len(diff)}")
    print(f"  Max Differenz:     {diff.max():.10f} px")
    print(f"  Mean Differenz:    {diff.mean():.10f} px")

    if diff.max() < 1e-6:
        print("  ✅ BESTANDEN – Manuelle und cv2 Projektion identisch")
    else:
        print("  ❌ FEHLER – Projektionen weichen ab!")


def test_rotation_valid(mesh_path, width, height, azimuth_deg, yfov):
    """
    TEST 4: Ist R eine gültige Rotationsmatrix?

    R^T @ R == I  und  det(R) == +1
    """
    print()
    print("=" * 60)
    print("TEST 4: Rotationsmatrix gültig?")
    print("=" * 60)

    cal = get_camera_calibration(mesh_path, width, height, azimuth_deg, yfov)
    R = cal["R"]

    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))
    det = np.linalg.det(R)

    print(f"  ||R^T R - I|| = {ortho_error:.2e}")
    print(f"  det(R)        = {det:.10f}")

    if ortho_error < 1e-10 and abs(det - 1.0) < 1e-10:
        print("  ✅ BESTANDEN – Gültige Rotationsmatrix")
    else:
        print("  ❌ FEHLER – Keine gültige Rotationsmatrix!")


def test_multi_azimuth(mesh_path, width, height, yfov, out_dir):
    """
    TEST 5: Rendere mehrere Blickwinkel und überlagere Vertices.
    """
    print()
    print("=" * 60)
    print("TEST 5: Mehrere Blickwinkel")
    print("=" * 60)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for az in [0, 45, 90, 135, 180, -90]:
        out_path = out_dir / f"verify_az{az:+04d}.png"

        color, _ = render_mesh(mesh_path, width, height, az, yfov)
        cal = get_camera_calibration(mesh_path, width, height, az, yfov)

        tri = trimesh.load(mesh_path, process=False)
        vertices = np.asarray(tri.vertices, dtype=np.float64)
        pts_2d, valid = project_vertices(vertices, cal["K"], cal["R"], cal["t"])

        img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        for pt in pts_2d:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        cv2.imwrite(str(out_path), img)
        print(f"  Azimuth {az:+4d}°: {out_path}")

    print("  ✅ Bilder gespeichert – visuell prüfen!")


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(description="Verify camera calibration")
    parser.add_argument("--mesh", required=True, help="Path to .obj mesh file")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--yfov", type=float, default=np.pi / 4.0)
    parser.add_argument("--out_dir", type=str, default="verify_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mesh:    {args.mesh}")
    print(f"Size:    {args.width}x{args.height}")
    print(f"Azimuth: {args.azimuth}°")
    print(f"yfov:    {args.yfov:.4f} rad")
    print()

    # Test 1: Visuell
    test_projection_on_image(
        args.mesh, args.width, args.height, args.azimuth, args.yfov,
        out_dir / "overlay.png",
    )

    # Test 2: Tiefe
    test_depth_consistency(
        args.mesh, args.width, args.height, args.azimuth, args.yfov,
    )

    # Test 3: cv2 vs manuell
    test_reprojection_identity(
        args.mesh, args.width, args.height, args.azimuth, args.yfov,
    )

    # Test 4: Rotation
    test_rotation_valid(
        args.mesh, args.width, args.height, args.azimuth, args.yfov,
    )

    # Test 5: Multi-Azimuth
    test_multi_azimuth(
        args.mesh, args.width, args.height, args.yfov, out_dir,
    )

    print()
    print("=" * 60)
    print("ALLE TESTS ABGESCHLOSSEN")
    print("=" * 60)


if __name__ == "__main__":
    main()