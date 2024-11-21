# ffmpeg

ffmpeg is a powerful tool for processing video and audio files. It is a powerful tool that can be used to convert video and audio files, create video and audio files, and more.

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