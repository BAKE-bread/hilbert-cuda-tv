# third_party/

This project uses the single-header **stb_image** / **stb_image_write**
libraries (public domain / MIT, by Sean Barrett) for PNG/JPG/BMP image I/O,
to avoid a hard OpenCV dependency.

**Both files are vendored in this directory** (`stb_image.h` v2.30,
`stb_image_write.h` v1.16) — no download needed for a normal build.
`include/utils/ImageIO.h` expects exactly these two filenames here, and
`CMakeLists.txt` already adds `third_party/` to the include path.

(Historical note: an earlier point in this project's development could
not fetch these files automatically — no network egress in the authoring
sandbox at the time — so this README used to ask you to download them
yourself. The user has since provided the real files directly and they
are committed here for real; if you're reading instructions elsewhere,
e.g. an old commit message or a stale comment, that still says to
download them, you can ignore that — they're already here.)

If you ever need to update them (e.g. a newer stb release), re-fetch
with:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -OutFile "third_party/stb_image.h"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" -OutFile "third_party/stb_image_write.h"
```

Or just open these two URLs in a browser and save them into this folder:
- https://github.com/nothings/stb/blob/master/stb_image.h
- https://github.com/nothings/stb/blob/master/stb_image_write.h

If you'd rather use OpenCV instead (e.g. you already have it installed),
build with `-DHCTV_USE_OPENCV=ON` and these two files won't be used at
all — see README.md at the project root for details.
