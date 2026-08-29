# Batch-mode KLayout render of a zoomed region of a routed GDS to PNG, with
# the real Sky130 layer properties (proper metal-layer colors, and OFF for
# fill/tap cells, which otherwise dominate the image -- 292,776 fill-cell
# instances vs 51,199 real std cells at TILE=8, per metrics.json). Without
# the .lyp, a close crop is just filler-cell tiling moire, not routing.
#
#   klayout -z -nc -r render_gds_crop.py -rd gds=<in.gds> -rd out=<out.png> \
#     -rd lyp=<sky130A.lyp> -rd x0=<um> -rd y0=<um> -rd x1=<um> -rd y1=<um> \
#     [-rd px=3000] [-rd hide=fill,tap]

import pya

px = int(globals().get("px", 3000))
x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
hide_terms = [s.strip().lower() for s in globals().get("hide", "fill,tap").split(",") if s.strip()]

app = pya.Application.instance()
mw = app.main_window()
mw.load_layout(gds, 0)
view = mw.current_view()

lyp = globals().get("lyp", "")
if lyp:
    view.load_layer_props(lyp)

# Hide layers whose name matches a hide term (case-insensitive substring).
it = view.begin_layers()
while not it.at_end():
    lp = it.current()
    name = (lp.name or "") + " " + str(lp.source)
    if any(t in name.lower() for t in hide_terms):
        lp2 = lp.dup()
        lp2.visible = False
        view.set_layer_properties(it, lp2)
    it.next()

view.zoom_box(pya.DBox(x0, y0, x1, y1))
view.save_image_with_options(out, px, px, 0, 3, 0, pya.DBox(), False)
print("wrote %s at %dx%d (3x oversampled), box (%.1f,%.1f)-(%.1f,%.1f)" %
      (out, px, px, x0, y0, x1, y1))
