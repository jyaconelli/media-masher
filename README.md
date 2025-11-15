# Media Masher
Creates crunchy images and videos by reconstructing reference media using only the exact pixels of input media.

# Quick Start

```bash
mkdir build && cd build

# Point CMake at Homebrew's OpenCV
cmake .. -DOpenCV_DIR="$(brew --prefix opencv)/share/opencv4"
make
```

# Usage
Use the `media_masher` binary built in `build/` after running CMake. Paths can be absolute or relative—just ensure OpenCV can read them.

## Command Options
```
[--image] <input_path> <ref_image_1[,ref_image_2...]> [additional_ref_images...] <output_file_path> <block_width> <block_height>
```

- Set `--image` if your input is an image, otherwise assumes video input
- `block_width` and `block_height` measured in pixels determine the sizes of the blocks of source and input images considered. `1 1` will check and pick each pixel independently.

## For videos
```bash
./build/media_masher \
  assets/source-video.mp4 \
  assets/1.jpeg,assets/2.jpeg,assets/3.jpeg \
  out/glitchy-video.mp4 \
  32 32
```
Or mix comma-separated and positional reference images:
```bash
./build/media_masher \
  assets/source-video.mp4 \
  assets/1.jpeg assets/2.jpeg assets/misc/*.png \
  out/glitchy-video.mp4 \
  16 12
```

## For images
```bash
./build/media_masher \
  --image \
  assets/input-still.png \
  assets/1.jpeg assets/2.jpeg \
  out/photomosaic.png \
  12 12
```
Block sizes control the granularity of the replacement tiles; smaller numbers mean more detail at the cost of runtime.

# Example Output
![example](https://github.com/user-attachments/assets/e9d1e207-0ee4-4ae4-8ed0-1cbf697d0321)
