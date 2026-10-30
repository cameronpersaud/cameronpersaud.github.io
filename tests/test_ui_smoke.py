"""Drive the whole app through all five steps without a screen and save screenshots.

python tests/test_ui_smoke.py <demo_folder> <screenshot_folder>
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from brainstack import io  # noqa: E402
from brainstack.ui.main_window import MainWindow, apply_dark_palette  # noqa: E402


def wait(win, app, timeout=900):
    t0 = time.time()
    while win.worker is not None:
        app.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout:
            raise TimeoutError
    for _ in range(20):
        app.processEvents()


def shot(win, folder, name):
    win.grab().save(os.path.join(folder, name))


def main(demo, out):
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv)
    apply_dark_palette(app)
    # Fail loudly instead of blocking on dialogs.
    QMessageBox.critical = staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError(a[2])))
    QMessageBox.warning = staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError(a[2])))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    win = MainWindow()
    win.resize(1400, 880)
    win.show()

    p0 = win.page_objs[0]
    p0._set_paths(io.find_images(demo))
    p0.res.setCurrentIndex(0)  # 512 px for speed
    assert "front" in p0.list.item(0).text() and "back" in p0.list.item(p0.list.count() - 1).text()
    print("List ends:", p0.list.item(0).text(), "|", p0.list.item(p0.list.count() - 1).text())
    app.processEvents()
    shot(win, out, "1_import.png")
    win.go_next(); wait(win, app)
    assert win.current() == 1, "did not reach calibration"

    p1 = win.page_objs[1]
    p1.z.setValue(100)
    p1._measure_tissue(); wait(win, app)
    shot(win, out, "2_calibrate.png")
    print("Calibration:", p1.build_calibration().describe())
    win.go_next(); wait(win, app)
    assert win.current() == 2

    p2 = win.page_objs[2]
    assert win.state.dye_channel == 2, f"dye channel guessed wrong: {win.state.dye_channel}"
    names = ["DAPI", "GFP", "Dextran"]
    for e, n in zip(p2._name_edits, names):
        e.setText(n)
    colors = [(90, 150, 255), (70, 255, 110), (255, 50, 50)]
    for i, (b, c) in enumerate(zip(p2._color_btns, colors)):
        b.set_color(c)
        p2._color_changed(i, c)
    print("Brain outline from channels:", p2.settings().mask_channel)
    p2._update_preview()
    shot(win, out, "3_prepare.png")
    win.go_next(); wait(win, app)
    assert win.current() == 3

    p3 = win.page_objs[3]
    p3.upright.setChecked(False)      # the demo stack has one slice mounted upside down
    p3._auto_align(); wait(win, app)
    p3.slider.setValue(10)
    app.processEvents()
    shot(win, out, "4_align.png")
    win.go_next(); wait(win, app)
    assert win.current() == 4, "did not reach 3D view"

    p4 = win.page_objs[4]
    wait(win, app)                                   # dye detection runs in the background
    for _ in range(30):
        app.processEvents()
    res = win.state.dye_result
    assert res is not None and res.found, "dye not found"
    print("Dye report:", p4.report.text()[:160].replace("<br>", " | "))
    shot(win, out, "5_dye_view.png")
    p4.plotter.screenshot(os.path.join(out, "5_canvas_dye_surface_solid.png"))
    # brain haze + dye glow
    p4.brain_style.setCurrentIndex(1)
    p4.dye_style.setCurrentIndex(1)
    p4.brain_opa.setValue(25)
    p4.render()
    p4.plotter.screenshot(os.path.join(out, "5_canvas_dye_haze_glow.png"))
    p4.brain_style.setCurrentIndex(2)
    p4.dye_style.setCurrentIndex(0)
    p4.render()
    p4.plotter.screenshot(os.path.join(out, "5_canvas_dye_slices_solid.png"))
    p4.brain_style.setCurrentIndex(0)
    # dense core preset
    p4.r_core.setChecked(True)
    p4._run_dye_detection(); wait(win, app)
    print("Core threshold:", win.state.dye_result.threshold,
          "volume %.3f mm3" % win.state.dye_result._report_cache["dye_volume_mm3"])
    p4.plotter.screenshot(os.path.join(out, "5_canvas_dye_core.png"))
    # manual level
    p4.dye_thr.setValue(40)
    p4._run_dye_detection(); wait(win, app)
    assert win.state.dye_result.threshold == 40
    p4.r_auto.setChecked(True)
    p4._run_dye_detection(); wait(win, app)
    p4._zoom_to_dye()
    p4.plotter.screenshot(os.path.join(out, "5_canvas_dye_zoom.png"))
    # other views still work
    for i, name in ((1, "planes"), (3, "volume"), (2, "points"), (4, "surface")):
        p4.mode_group.button(i).setChecked(True)
        p4.render()
        p4.plotter.screenshot(os.path.join(out, f"5_canvas_{name}.png"))
    p4.mode_group.button(0).setChecked(True)
    p4.render()
    # exports
    from brainstack import dye as D
    from brainstack import volume as V
    D.export_dye_csv(win.state.vol, win.state.dye_result, os.path.join(out, "dye.csv"))
    win.state.dye_result._surface_cache.save(os.path.join(out, "dye.stl"))
    V.export_measurements_csv(win.state.vol, [60, 60, 60], os.path.join(out, "m.csv"))
    V.export_tiff(win.state.vol, os.path.join(out, "stack.tif"))
    V.export_surface(win.state.vol, os.path.join(out, "surface.stl"))
    p4.clip_btn.setChecked(True)
    p4.clip_btn.setChecked(False)
    p4.dist_btn.setChecked(True)
    p4._on_pick((0, 0, 0))
    p4._on_pick((3000, 4000, 0))
    print("Distance test:", p4.distances)
    p4.dist_btn.setChecked(False)

    # ---- anatomical report, then flip the file order and check it mirrors
    from brainstack.orientation import Orientation, anatomical_position
    rep1 = win.state.dye_result._report_cache
    an1 = anatomical_position(rep1, win.state.orientation)
    print(f"Coronal: dye {an1['from_front_um']:.0f} um behind front, {an1['from_top_um']:.0f} below "
          f"top, {an1['midline_offset_um']:.0f} {an1['midline_side']} of midline")
    shot(win, out, "5_dye_report.png")
    for plane, cam in (("coronal", "3d"), ("sagittal", "3d"), ("transverse", "3d")):
        win.state.orientation = Orientation(plane)
        p4.render(); p4._camera(cam)
        p4.plotter.screenshot(os.path.join(out, f"5_plane_{plane}.png"))
    win.state.orientation = Orientation("coronal")
    p4.render()
    p4._camera("front"); p4.plotter.screenshot(os.path.join(out, "5_cam_front.png"))
    p4._camera("top"); p4.plotter.screenshot(os.path.join(out, "5_cam_top.png"))
    p4._camera("side"); p4.plotter.screenshot(os.path.join(out, "5_cam_side.png"))

    t0 = time.time()
    win._show_step(0)
    p0.orient.order.setCurrentIndex(1)               # "my files are numbered back -> front"
    assert "front" in p0.list.item(0).text()          # list re-ordered so slice 1 is still the front
    assert os.path.basename(p0.state.paths[0]) == "slice_030.tif", p0.state.paths[0]
    for _ in range(4):
        win.go_next(); wait(win, app)
    wait(win, app)
    assert win.current() == 4
    rep2 = win.state.dye_result._report_cache
    an2 = anatomical_position(rep2, win.state.orientation)
    print(f"After flipping order ({time.time() - t0:.1f}s, no reload): dye "
          f"{an2['from_front_um']:.0f} um behind front (was {an1['from_front_um']:.0f}; "
          f"brain length {an2['brain_length_um']:.0f})")
    assert abs((an1["from_front_um"] + an2["from_front_um"]) - an2["brain_length_um"]) < 150
    assert abs(an1["from_top_um"] - an2["from_top_um"]) < 60

    # ---- straighten whole stack: +90 then -90 returns to the same transforms
    p3 = win.page_objs[3]
    before = [(t.shift_x, t.shift_y, t.angle_deg) for t in win.state.transforms]
    p3._rotate_all(90); p3._rotate_all(-90)
    after = [(t.shift_x, t.shift_y, t.angle_deg) for t in win.state.transforms]
    assert np.allclose(before, after, atol=1e-6)
    print("Rotate-whole-stack round trip OK")
    print("UI smoke test passed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
