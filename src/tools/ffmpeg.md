# ffmpeg

## Convert video to mp4

```bash
ffmpeg -i input.mp4 -c:v h264 -c:a aac output.mp4
```

## HLS

```bash
ffmpeg -i input.mp4 -hls_time 10 -hls_list_size 0 -f hls output.m3u8
```

## Screenshot

```bash
ffmpeg -i input.mp4 -vf "select=eq(n\,0)" -q:v 1 -f image2 output.png
```