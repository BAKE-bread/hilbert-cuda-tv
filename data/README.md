# data/

Place input images here if you don't want to pass full paths on the
command line, e.g.:

```powershell
.\HilbertCUDA-TV.exe --input data\my_photo.png --output output\result.png
```

This directory is otherwise empty by default — the project doesn't ship
with any bundled test images (see README.md's note on why the classic
"Lena" test image specifically was avoided: licensing ambiguity. The
`--demo` flag generates a synthetic test image in-process instead, so you
don't need anything here to try the tool out).
