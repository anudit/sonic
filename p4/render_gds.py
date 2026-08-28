# Batch-mode KLayout render of a routed GDS to PNG.
#
#   klayout -z -nc -r render_gds.py -rd gds=<in.gds> -rd out=<out.png> [-rd size=4000]
#
# -z gives a hidden main window, which still owns a LayoutView, which is what
# save_image needs. Run under xvfb where no display exists at all.

import pya

size = int(globals().get("size", 4000))

app = pya.Application.instance()
mw = app.main_window()

mw.load_layout(gds, 0)
view = mw.current_view()

view.max_hier()
view.zoom_fit()
view.save_image(out, size, size)

print("wrote %s at %dx%d" % (out, size, size))
