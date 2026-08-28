# demo

`floorplan.html` — an interactive, area-accurate floorplan of Sonic S1.

Every rectangle is sized by its real area: `data.json` is generated from
`sonic/roofline.area`, and the page lays it out with a squarified treemap, so a
block's share of the picture is its share of the die. Switching SKU re-runs the
same model, which is why the die, the PHY and the SRAM all change together.

Regenerate the data after any change to `sonic/roofline.py` or `chipspec.py`:

```
make demo-data      # rewrites demo/data.json and re-injects it
python3 -m http.server -d demo 8731    # then open localhost:8731/floorplan.html
```

Opening the file over `file://` works in most browsers, but a local server
avoids the font stylesheet being blocked.
