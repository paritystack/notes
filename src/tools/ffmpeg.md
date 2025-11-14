# FFmpeg

FFmpeg is a complete, cross-platform solution to record, convert, and stream audio and video. It's one of the most powerful multimedia frameworks available, supporting virtually every codec and format.

## Overview

FFmpeg is a command-line tool that can handle virtually any multimedia processing task. It consists of several components including ffmpeg (transcoder), ffprobe (media analyzer), and ffplay (media player).

**Key Features:**
- Convert between virtually all audio/video formats
- Change codecs, bitrates, and quality settings
- Extract audio from video or vice versa
- Resize, crop, rotate, and flip videos
- Apply filters and effects
- Generate thumbnails and screenshots
- Concatenate multiple files
- Stream to various protocols (RTMP, HLS, DASH)
- Hardware acceleration support
- Subtitle handling (extract, embed, burn-in)

**Components:**
- **ffmpeg**: Main command-line tool for conversion and processing
- **ffprobe**: Analyze media files (metadata, streams, format)
- **ffplay**: Simple media player for testing
- **libavcodec**: Codec library
- **libavformat**: Container format library
- **libavfilter**: Audio/video filtering library

---

## Installation

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Fedora/RHEL
sudo dnf install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# Build from source (latest features)
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
./configure --enable-gpl --enable-libx264 --enable-libx265
make
sudo make install
```

### macOS

```bash
# Using Homebrew
brew install ffmpeg

# With additional codecs
brew install ffmpeg --with-libvpx --with-libvorbis --with-x265

# Check version
ffmpeg -version
```

### Windows

```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Extract and add to PATH
```

---

## Basic Concepts

### Containers vs Codecs

- **Container** (format): Wrapper that holds audio/video/subtitle streams (e.g., MP4, MKV, AVI)
- **Codec**: Algorithm for encoding/decoding media (e.g., H.264, AAC, VP9)

Common combinations:
- MP4 container: H.264 video + AAC audio
- MKV container: H.265 video + Opus audio
- WebM container: VP9 video + Vorbis audio

### Stream Selection

FFmpeg identifies streams as:
- `0:v:0` - First video stream
- `0:a:0` - First audio stream
- `0:s:0` - First subtitle stream

### Common Codec Identifiers

**Video:**
- `libx264` - H.264/AVC (widely compatible)
- `libx265` - H.265/HEVC (better compression)
- `libvpx-vp9` - VP9 (open, good for web)
- `libaom-av1` - AV1 (newest, best compression)

**Audio:**
- `aac` - AAC (standard)
- `libmp3lame` - MP3
- `libopus` - Opus (best quality/size)
- `libvorbis` - Vorbis (open)

---

## Basic Usage

### Get Media Information

```bash
# Detailed file information
ffprobe input.mp4

# Show only format information
ffprobe -show_format input.mp4

# Show stream information
ffprobe -show_streams input.mp4

# JSON output
ffprobe -print_format json -show_format -show_streams input.mp4

# Get video duration
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 input.mp4

# Get video resolution
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 input.mp4

# Get video framerate
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 input.mp4

# Get bitrate
ffprobe -v error -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 input.mp4
```

### Simple Conversion

```bash
# Basic format conversion (auto-detect codecs)
ffmpeg -i input.avi output.mp4

# Convert with progress
ffmpeg -i input.avi -progress - output.mp4

# Overwrite output without prompt
ffmpeg -y -i input.avi output.mp4

# Never overwrite
ffmpeg -n -i input.avi output.mp4
```

---

## Video Conversion

### Format Conversion

```bash
# AVI to MP4
ffmpeg -i input.avi output.mp4

# MKV to MP4
ffmpeg -i input.mkv -c copy output.mp4  # Copy streams (fast)

# MOV to MP4
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4

# WebM to MP4
ffmpeg -i input.webm -c:v libx264 -c:a aac output.mp4

# FLV to MP4
ffmpeg -i input.flv -c:v libx264 -c:a aac output.mp4

# MP4 to WebM
ffmpeg -i input.mp4 -c:v libvpx-vp9 -c:a libopus output.webm

# Any format to GIF
ffmpeg -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" output.gif
```

### Stream Copying (Fast)

```bash
# Copy all streams without re-encoding
ffmpeg -i input.mp4 -c copy output.mkv

# Copy video, re-encode audio
ffmpeg -i input.mp4 -c:v copy -c:a aac output.mp4

# Copy audio, re-encode video
ffmpeg -i input.mp4 -c:v libx264 -c:a copy output.mp4
```

---

## Video Encoding

### H.264 Encoding

```bash
# Basic H.264 encoding
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4

# High quality H.264
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k output.mp4

# Web-optimized H.264
ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 22 -c:a aac -b:a 128k -movflags +faststart output.mp4

# Specific bitrate
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -c:a aac -b:a 128k output.mp4

# Two-pass encoding (better quality)
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 1 -f mp4 /dev/null
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 2 output.mp4

# Presets (speed vs compression)
# ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 20 output.mp4

# Profiles and levels
ffmpeg -i input.mp4 -c:v libx264 -profile:v baseline -level 3.0 output.mp4
ffmpeg -i input.mp4 -c:v libx264 -profile:v main -level 4.0 output.mp4
ffmpeg -i input.mp4 -c:v libx264 -profile:v high -level 4.2 output.mp4
```

### H.265/HEVC Encoding

```bash
# Basic H.265 encoding
ffmpeg -i input.mp4 -c:v libx265 -c:a aac output.mp4

# High quality H.265
ffmpeg -i input.mp4 -c:v libx265 -preset slow -crf 22 -c:a aac output.mp4

# 4K H.265
ffmpeg -i input.mp4 -c:v libx265 -preset medium -crf 24 -c:a aac -tag:v hvc1 output.mp4

# H.265 with specific bitrate
ffmpeg -i input.mp4 -c:v libx265 -b:v 1.5M -c:a aac output.mp4
```

### VP9 Encoding (WebM)

```bash
# Basic VP9
ffmpeg -i input.mp4 -c:v libvpx-vp9 -c:a libopus output.webm

# High quality VP9
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -c:a libopus output.webm

# VP9 two-pass
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 1M -pass 1 -f webm /dev/null
ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 1M -pass 2 -c:a libopus output.webm

# VP9 with quality settings
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -c:a libopus -b:a 128k output.webm
```

### AV1 Encoding

```bash
# Basic AV1 (slow but best compression)
ffmpeg -i input.mp4 -c:v libaom-av1 -crf 30 -c:a libopus output.webm

# AV1 with speed settings
ffmpeg -i input.mp4 -c:v libaom-av1 -cpu-used 4 -crf 30 output.webm

# SVT-AV1 (faster)
ffmpeg -i input.mp4 -c:v libsvtav1 -crf 35 -c:a libopus output.webm
```

### Quality Control

```bash
# CRF (Constant Rate Factor) - recommended
# Lower = better quality, larger file
# H.264: 18-28 (23 default)
# H.265: 22-32 (28 default)
# VP9: 15-35 (30 default)
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4

# CBR (Constant Bitrate)
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -minrate 2M -maxrate 2M -bufsize 1M output.mp4

# VBR (Variable Bitrate)
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -maxrate 3M -bufsize 2M output.mp4

# Target file size
# Calculate bitrate: (target_size_MB * 8192) / duration_seconds
ffmpeg -i input.mp4 -c:v libx264 -b:v 1500k -pass 1 -f mp4 /dev/null
ffmpeg -i input.mp4 -c:v libx264 -b:v 1500k -pass 2 output.mp4
```

---

## Audio Operations

### Audio Extraction

```bash
# Extract audio to MP3
ffmpeg -i input.mp4 -vn -c:a libmp3lame -b:a 192k output.mp3

# Extract audio to AAC
ffmpeg -i input.mp4 -vn -c:a aac -b:a 192k output.aac

# Extract audio to FLAC (lossless)
ffmpeg -i input.mp4 -vn -c:a flac output.flac

# Extract audio without re-encoding
ffmpeg -i input.mp4 -vn -c:a copy output.aac
```

### Audio Conversion

```bash
# Convert audio format
ffmpeg -i input.mp3 output.wav
ffmpeg -i input.wav -c:a libmp3lame -b:a 320k output.mp3
ffmpeg -i input.mp3 -c:a aac -b:a 192k output.aac
ffmpeg -i input.wav -c:a libopus -b:a 128k output.opus

# Change sample rate
ffmpeg -i input.mp3 -ar 44100 output.mp3
ffmpeg -i input.wav -ar 48000 output.wav

# Change channels (mono/stereo)
ffmpeg -i input.mp3 -ac 1 output.mp3  # Mono
ffmpeg -i input.mp3 -ac 2 output.mp3  # Stereo

# Normalize audio
ffmpeg -i input.mp3 -af "loudnorm" output.mp3

# Change volume
ffmpeg -i input.mp3 -af "volume=2.0" output.mp3  # Double volume
ffmpeg -i input.mp3 -af "volume=0.5" output.mp3  # Half volume
ffmpeg -i input.mp3 -af "volume=10dB" output.mp3  # Increase by 10dB
```

### Audio Bitrate

```bash
# Constant bitrate
ffmpeg -i input.mp4 -c:a aac -b:a 128k output.mp4

# Common bitrates
ffmpeg -i input.mp4 -c:a aac -b:a 96k output.mp4   # Low quality
ffmpeg -i input.mp4 -c:a aac -b:a 128k output.mp4  # Standard
ffmpeg -i input.mp4 -c:a aac -b:a 192k output.mp4  # Good quality
ffmpeg -i input.mp4 -c:a aac -b:a 256k output.mp4  # High quality
ffmpeg -i input.mp4 -c:a aac -b:a 320k output.mp4  # Maximum quality
```

### Merge Audio and Video

```bash
# Replace audio in video
ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output.mp4

# Add audio track (multiple audio streams)
ffmpeg -i video.mp4 -i audio.mp3 -c copy -map 0 -map 1:a output.mp4

# Mix two audio tracks
ffmpeg -i input1.mp3 -i input2.mp3 -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest" output.mp3
```

---

## Video Filters

### Resize and Scale

```bash
# Resize to specific dimensions
ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4

# Resize maintaining aspect ratio
ffmpeg -i input.mp4 -vf "scale=1280:-1" output.mp4  # Width 1280, auto height
ffmpeg -i input.mp4 -vf "scale=-1:720" output.mp4   # Height 720, auto width

# Scale to percentage
ffmpeg -i input.mp4 -vf "scale=iw*0.5:ih*0.5" output.mp4  # 50% size

# Common resolutions
ffmpeg -i input.mp4 -vf "scale=1920:1080" output.mp4  # 1080p
ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4   # 720p
ffmpeg -i input.mp4 -vf "scale=854:480" output.mp4    # 480p
ffmpeg -i input.mp4 -vf "scale=640:360" output.mp4    # 360p

# High quality scaling
ffmpeg -i input.mp4 -vf "scale=1920:1080:flags=lanczos" output.mp4
```

### Crop

```bash
# Crop to specific size
# crop=width:height:x:y
ffmpeg -i input.mp4 -vf "crop=1280:720:0:0" output.mp4

# Crop center
ffmpeg -i input.mp4 -vf "crop=1920:800:0:140" output.mp4

# Crop to 16:9 from 4:3
ffmpeg -i input.mp4 -vf "crop=in_h*16/9:in_h" output.mp4

# Auto-detect crop
ffmpeg -i input.mp4 -vf "cropdetect" -f null -
# Then use detected values
ffmpeg -i input.mp4 -vf "crop=1920:800:0:140" output.mp4

# Crop and scale
ffmpeg -i input.mp4 -vf "crop=1920:800:0:140,scale=1280:534" output.mp4
```

### Rotate and Flip

```bash
# Rotate 90 degrees clockwise
ffmpeg -i input.mp4 -vf "transpose=1" output.mp4

# Rotate 90 degrees counter-clockwise
ffmpeg -i input.mp4 -vf "transpose=2" output.mp4

# Rotate 180 degrees
ffmpeg -i input.mp4 -vf "transpose=2,transpose=2" output.mp4

# Flip horizontal
ffmpeg -i input.mp4 -vf "hflip" output.mp4

# Flip vertical
ffmpeg -i input.mp4 -vf "vflip" output.mp4

# Rotate by arbitrary angle
ffmpeg -i input.mp4 -vf "rotate=45*PI/180" output.mp4
```

### Watermark

```bash
# Add image watermark
ffmpeg -i input.mp4 -i logo.png -filter_complex "overlay=10:10" output.mp4

# Watermark in bottom right
ffmpeg -i input.mp4 -i logo.png -filter_complex "overlay=W-w-10:H-h-10" output.mp4

# Watermark centered
ffmpeg -i input.mp4 -i logo.png -filter_complex "overlay=(W-w)/2:(H-h)/2" output.mp4

# Transparent watermark
ffmpeg -i input.mp4 -i logo.png -filter_complex "[1:v]format=rgba,colorchannelmixer=aa=0.5[logo];[0:v][logo]overlay=10:10" output.mp4

# Text watermark
ffmpeg -i input.mp4 -vf "drawtext=text='Copyright 2024':x=10:y=10:fontsize=24:fontcolor=white" output.mp4

# Text with shadow
ffmpeg -i input.mp4 -vf "drawtext=text='Copyright':x=10:y=10:fontsize=36:fontcolor=white:shadowcolor=black:shadowx=2:shadowy=2" output.mp4

# Dynamic timestamp
ffmpeg -i input.mp4 -vf "drawtext=text='%{localtime\:%Y-%m-%d %H\\:%M\\:%S}':x=10:y=10:fontsize=24:fontcolor=white" output.mp4
```

### Fade In/Out

```bash
# Fade in video (first 2 seconds)
ffmpeg -i input.mp4 -vf "fade=in:0:60" output.mp4

# Fade out video (last 2 seconds)
ffmpeg -i input.mp4 -vf "fade=out:st=28:d=2" output.mp4

# Fade in and out
ffmpeg -i input.mp4 -vf "fade=in:0:60,fade=out:st=28:d=2" output.mp4

# Audio fade in/out
ffmpeg -i input.mp4 -af "afade=in:st=0:d=2,afade=out:st=28:d=2" output.mp4

# Combined video and audio fade
ffmpeg -i input.mp4 -vf "fade=in:0:60,fade=out:st=28:d=60" -af "afade=in:st=0:d=2,afade=out:st=28:d=2" output.mp4
```

### Color Adjustments

```bash
# Brightness
ffmpeg -i input.mp4 -vf "eq=brightness=0.1" output.mp4

# Contrast
ffmpeg -i input.mp4 -vf "eq=contrast=1.5" output.mp4

# Saturation
ffmpeg -i input.mp4 -vf "eq=saturation=1.5" output.mp4

# Gamma
ffmpeg -i input.mp4 -vf "eq=gamma=1.2" output.mp4

# Combined adjustments
ffmpeg -i input.mp4 -vf "eq=brightness=0.1:contrast=1.2:saturation=1.3" output.mp4

# Grayscale
ffmpeg -i input.mp4 -vf "hue=s=0" output.mp4

# Sepia tone
ffmpeg -i input.mp4 -vf "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131" output.mp4
```

### Blur and Sharpen

```bash
# Blur
ffmpeg -i input.mp4 -vf "boxblur=5:1" output.mp4

# Gaussian blur
ffmpeg -i input.mp4 -vf "gblur=sigma=5" output.mp4

# Sharpen
ffmpeg -i input.mp4 -vf "unsharp=5:5:1.5:5:5:0.0" output.mp4

# Denoise
ffmpeg -i input.mp4 -vf "nlmeans" output.mp4
```

---

## Advanced Filters

### Complex Filter Chains

```bash
# Scale and crop
ffmpeg -i input.mp4 -vf "scale=1920:1080,crop=1920:800:0:140" output.mp4

# Multiple filters
ffmpeg -i input.mp4 -vf "scale=1280:720,hue=s=1.5,eq=brightness=0.1" output.mp4

# Filter with audio
ffmpeg -i input.mp4 -vf "scale=1280:720" -af "volume=2.0" output.mp4
```

### Picture-in-Picture

```bash
# Basic PIP
ffmpeg -i main.mp4 -i overlay.mp4 -filter_complex \
  "[1:v]scale=320:240[pip];[0:v][pip]overlay=W-w-10:H-h-10" \
  output.mp4

# PIP with different positions
# Top-left
ffmpeg -i main.mp4 -i overlay.mp4 -filter_complex \
  "[1:v]scale=320:240[pip];[0:v][pip]overlay=10:10" output.mp4

# Top-right
ffmpeg -i main.mp4 -i overlay.mp4 -filter_complex \
  "[1:v]scale=320:240[pip];[0:v][pip]overlay=W-w-10:10" output.mp4

# Bottom-left
ffmpeg -i main.mp4 -i overlay.mp4 -filter_complex \
  "[1:v]scale=320:240[pip];[0:v][pip]overlay=10:H-h-10" output.mp4
```

### Side-by-Side

```bash
# Side-by-side comparison
ffmpeg -i left.mp4 -i right.mp4 -filter_complex \
  "[0:v][1:v]hstack=inputs=2" output.mp4

# Vertical stack
ffmpeg -i top.mp4 -i bottom.mp4 -filter_complex \
  "[0:v][1:v]vstack=inputs=2" output.mp4

# 2x2 grid
ffmpeg -i input1.mp4 -i input2.mp4 -i input3.mp4 -i input4.mp4 \
  -filter_complex \
  "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack" \
  output.mp4
```

### Speed Changes

```bash
# Speed up video (2x)
ffmpeg -i input.mp4 -vf "setpts=0.5*PTS" output.mp4

# Slow down video (0.5x)
ffmpeg -i input.mp4 -vf "setpts=2.0*PTS" output.mp4

# Speed up audio
ffmpeg -i input.mp4 -filter:a "atempo=2.0" output.mp4

# Speed up both video and audio (2x)
ffmpeg -i input.mp4 -vf "setpts=0.5*PTS" -af "atempo=2.0" output.mp4

# Slow motion (0.5x) with audio
ffmpeg -i input.mp4 -vf "setpts=2.0*PTS" -af "atempo=0.5" output.mp4

# Speed limits: atempo must be between 0.5 and 2.0
# For 4x speed, chain multiple atempo filters
ffmpeg -i input.mp4 -filter:a "atempo=2.0,atempo=2.0" output.mp4
```

### Framerate Changes

```bash
# Change framerate
ffmpeg -i input.mp4 -r 30 output.mp4  # 30 fps
ffmpeg -i input.mp4 -r 60 output.mp4  # 60 fps

# Convert to 24fps (film)
ffmpeg -i input.mp4 -r 24 output.mp4

# Duplicate frames to increase fps
ffmpeg -i input.mp4 -vf "fps=60" output.mp4

# Interpolate frames (smooth)
ffmpeg -i input.mp4 -vf "minterpolate=fps=60:mi_mode=mci" output.mp4
```

---

## Screenshots and Thumbnails

### Extract Single Frame

```bash
# Extract first frame
ffmpeg -i input.mp4 -vf "select=eq(n\,0)" -q:v 1 -frames:v 1 output.png

# Extract frame at specific time
ffmpeg -ss 00:00:10 -i input.mp4 -frames:v 1 output.jpg

# Extract frame at 5 seconds
ffmpeg -ss 5 -i input.mp4 -frames:v 1 output.png

# High quality screenshot
ffmpeg -ss 00:01:30 -i input.mp4 -frames:v 1 -q:v 2 output.jpg

# Specific size screenshot
ffmpeg -ss 10 -i input.mp4 -vf "scale=1920:1080" -frames:v 1 output.png
```

### Extract Multiple Frames

```bash
# Extract every frame
ffmpeg -i input.mp4 frame_%04d.png

# Extract 1 frame per second
ffmpeg -i input.mp4 -vf "fps=1" frame_%04d.png

# Extract 1 frame every 10 seconds
ffmpeg -i input.mp4 -vf "fps=1/10" frame_%04d.png

# Extract frames from specific time range
ffmpeg -ss 00:00:10 -t 00:00:05 -i input.mp4 -vf "fps=1" frame_%04d.png

# Extract frames with specific quality
ffmpeg -i input.mp4 -vf "fps=1" -q:v 2 frame_%04d.jpg
```

### Create Thumbnails

```bash
# Create thumbnail grid (contact sheet)
ffmpeg -i input.mp4 -vf "fps=1/60,scale=320:240,tile=4x3" thumbnail.png

# Create thumbnail at specific interval
ffmpeg -i input.mp4 -vf "thumbnail=300" -frames:v 1 thumb.png

# Create multiple thumbnails
ffmpeg -i input.mp4 -vf "fps=1/60" thumb_%03d.jpg
```

### Create GIF

```bash
# Basic GIF
ffmpeg -i input.mp4 output.gif

# High quality GIF
ffmpeg -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output.gif

# GIF from specific time range
ffmpeg -ss 5 -t 10 -i input.mp4 -vf "fps=10,scale=480:-1:flags=lanczos" output.gif

# Optimized GIF with custom palette
ffmpeg -i input.mp4 -vf "fps=15,scale=480:-1:flags=lanczos,palettegen" palette.png
ffmpeg -i input.mp4 -i palette.png -filter_complex "fps=15,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse" output.gif
```

---

## Concatenation and Trimming

### Trim/Cut Video

```bash
# Cut from start time for duration
ffmpeg -ss 00:00:10 -t 00:00:30 -i input.mp4 -c copy output.mp4

# Cut from start to end time
ffmpeg -ss 00:00:10 -to 00:00:40 -i input.mp4 -c copy output.mp4

# Cut with re-encoding (more precise)
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:30 -c:v libx264 -c:a aac output.mp4

# Multiple segments
ffmpeg -i input.mp4 -ss 00:00:00 -t 00:00:10 part1.mp4
ffmpeg -i input.mp4 -ss 00:00:10 -t 00:00:10 part2.mp4
ffmpeg -i input.mp4 -ss 00:00:20 -t 00:00:10 part3.mp4

# Cut last N seconds
ffmpeg -sseof -10 -i input.mp4 -c copy last_10sec.mp4
```

### Concatenate Videos

```bash
# Method 1: Concat demuxer (same codec, fast)
# Create file list
echo "file 'video1.mp4'" > filelist.txt
echo "file 'video2.mp4'" >> filelist.txt
echo "file 'video3.mp4'" >> filelist.txt

ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4

# Method 2: Concat filter (different codecs)
ffmpeg -i video1.mp4 -i video2.mp4 -i video3.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[outv][outa]" \
  -map "[outv]" -map "[outa]" output.mp4

# Method 3: Concat protocol (identical files)
ffmpeg -i "concat:video1.mp4|video2.mp4|video3.mp4" -c copy output.mp4

# Concatenate with transition
ffmpeg -i input1.mp4 -i input2.mp4 -filter_complex \
  "[0:v]fade=out:st=9:d=1[v0];[1:v]fade=in:st=0:d=1[v1];[v0][v1]concat=n=2:v=1:a=0" \
  output.mp4
```

### Split Video

```bash
# Split into equal parts
ffmpeg -i input.mp4 -c copy -map 0 -segment_time 300 -f segment output%03d.mp4

# Split by size
ffmpeg -i input.mp4 -c copy -map 0 -segment_size 100M -f segment output%03d.mp4

# Split at keyframes
ffmpeg -i input.mp4 -c copy -segment_time 300 -reset_timestamps 1 -f segment output%03d.mp4
```

---

## Streaming

### HLS (HTTP Live Streaming)

```bash
# Basic HLS
ffmpeg -i input.mp4 -hls_time 10 -hls_list_size 0 -f hls output.m3u8

# HLS with different quality levels (adaptive streaming)
ffmpeg -i input.mp4 \
  -vf "scale=1280:720" -c:v libx264 -b:v 2M -c:a aac -b:a 128k -hls_time 10 720p.m3u8 \
  -vf "scale=854:480" -c:v libx264 -b:v 1M -c:a aac -b:a 96k -hls_time 10 480p.m3u8 \
  -vf "scale=640:360" -c:v libx264 -b:v 500k -c:a aac -b:a 64k -hls_time 10 360p.m3u8

# HLS with segment naming
ffmpeg -i input.mp4 \
  -hls_time 10 \
  -hls_list_size 0 \
  -hls_segment_filename "segment_%03d.ts" \
  -f hls output.m3u8

# HLS with encryption
ffmpeg -i input.mp4 \
  -hls_time 10 \
  -hls_key_info_file key_info.txt \
  -hls_list_size 0 \
  -f hls output.m3u8

# HLS options
ffmpeg -i input.mp4 \
  -c:v libx264 -c:a aac \
  -hls_time 6 \                    # Segment duration
  -hls_list_size 0 \                # Keep all segments in playlist
  -hls_segment_type mpegts \        # Segment format
  -hls_flags delete_segments \      # Delete old segments
  -hls_start_number_source datetime \
  -f hls output.m3u8
```

### DASH (Dynamic Adaptive Streaming over HTTP)

```bash
# Basic DASH
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -f dash output.mpd

# DASH with multiple qualities
ffmpeg -i input.mp4 \
  -map 0:v -map 0:a -c:v libx264 -c:a aac \
  -b:v:0 2M -s:v:0 1280x720 \
  -b:v:1 1M -s:v:1 854x480 \
  -b:v:2 500k -s:v:3 640x360 \
  -adaptation_sets "id=0,streams=v id=1,streams=a" \
  -f dash output.mpd
```

### RTMP Streaming

```bash
# Stream to RTMP server
ffmpeg -re -i input.mp4 -c:v libx264 -preset veryfast -maxrate 3M \
  -bufsize 6M -c:a aac -b:a 128k -f flv rtmp://server/live/stream

# Stream with specific resolution and framerate
ffmpeg -re -i input.mp4 \
  -vf "scale=1280:720" -r 30 \
  -c:v libx264 -preset veryfast -b:v 2M \
  -c:a aac -b:a 128k \
  -f flv rtmp://server/live/stream

# Stream from webcam
ffmpeg -f v4l2 -i /dev/video0 -f alsa -i default \
  -c:v libx264 -preset veryfast -b:v 1M \
  -c:a aac -b:a 128k \
  -f flv rtmp://server/live/stream

# Re-stream (relay)
ffmpeg -i rtmp://source/live/stream -c copy -f flv rtmp://destination/live/stream
```

### UDP/RTP Streaming

```bash
# UDP streaming
ffmpeg -re -i input.mp4 -c:v libx264 -c:a aac -f mpegts udp://192.168.1.100:1234

# RTP streaming
ffmpeg -re -i input.mp4 -c:v libx264 -c:a aac -f rtp rtp://192.168.1.100:1234

# SRT streaming
ffmpeg -re -i input.mp4 -c:v libx264 -c:a aac -f mpegts srt://192.168.1.100:1234
```

---

## Subtitles

### Extract Subtitles

```bash
# Extract all subtitle tracks
ffmpeg -i input.mkv -c:s copy subtitles.srt

# Extract specific subtitle
ffmpeg -i input.mkv -map 0:s:0 -c:s copy subtitle_track1.srt

# Convert subtitle format
ffmpeg -i input.srt output.ass
ffmpeg -i input.ass output.srt
```

### Add Subtitles

```bash
# Soft subtitles (embedded, can be toggled)
ffmpeg -i input.mp4 -i subtitles.srt -c copy -c:s mov_text output.mp4

# Add multiple subtitle tracks
ffmpeg -i input.mp4 -i eng.srt -i spa.srt \
  -c copy -c:s mov_text \
  -metadata:s:s:0 language=eng \
  -metadata:s:s:1 language=spa \
  output.mp4

# Hard subtitles (burned in, always visible)
ffmpeg -i input.mp4 -vf "subtitles=subtitles.srt" output.mp4

# Burn subtitles with style
ffmpeg -i input.mp4 -vf "subtitles=subtitles.srt:force_style='FontName=Arial,FontSize=24,PrimaryColour=&H00FFFF'" output.mp4

# Burn ASS/SSA subtitles
ffmpeg -i input.mp4 -vf "ass=subtitles.ass" output.mp4
```

### Create Subtitles

```bash
# Generate subtitle from text file
# Create subtitle.srt:
# 1
# 00:00:00,000 --> 00:00:05,000
# First subtitle text
#
# 2
# 00:00:05,000 --> 00:00:10,000
# Second subtitle text

ffmpeg -i input.mp4 -i subtitle.srt -c copy -c:s mov_text output.mp4
```

---

## Metadata

### View Metadata

```bash
# Show all metadata
ffprobe -show_format -show_streams input.mp4

# Show only metadata
ffmpeg -i input.mp4 -f ffmetadata metadata.txt

# Extract cover art
ffmpeg -i input.mp3 -an -vcodec copy cover.jpg
```

### Edit Metadata

```bash
# Set metadata tags
ffmpeg -i input.mp4 -metadata title="My Video" \
  -metadata author="John Doe" \
  -metadata copyright="2024" \
  -c copy output.mp4

# Remove all metadata
ffmpeg -i input.mp4 -map_metadata -1 -c copy output.mp4

# Add cover art to audio
ffmpeg -i input.mp3 -i cover.jpg \
  -map 0:a -map 1:v \
  -c:a copy -c:v copy \
  -metadata:s:v title="Album cover" \
  -metadata:s:v comment="Cover (front)" \
  output.mp3

# Copy metadata from one file to another
ffmpeg -i source.mp4 -i destination.mp4 -map 1 -map_metadata 0 -c copy output.mp4
```

---

## Performance and Hardware Acceleration

### Hardware Encoding

```bash
# NVIDIA NVENC (H.264)
ffmpeg -i input.mp4 -c:v h264_nvenc -preset slow -b:v 2M output.mp4

# NVIDIA NVENC (H.265)
ffmpeg -i input.mp4 -c:v hevc_nvenc -preset slow -b:v 2M output.mp4

# Intel Quick Sync (H.264)
ffmpeg -i input.mp4 -c:v h264_qsv -preset slow -b:v 2M output.mp4

# Intel Quick Sync (H.265)
ffmpeg -i input.mp4 -c:v hevc_qsv -preset slow -b:v 2M output.mp4

# AMD VCE (H.264)
ffmpeg -i input.mp4 -c:v h264_amf -b:v 2M output.mp4

# Apple VideoToolbox (H.264)
ffmpeg -i input.mp4 -c:v h264_videotoolbox -b:v 2M output.mp4

# VA-API (Linux)
ffmpeg -vaapi_device /dev/dri/renderD128 -i input.mp4 \
  -vf 'format=nv12,hwupload' -c:v h264_vaapi -b:v 2M output.mp4
```

### Hardware Decoding

```bash
# NVIDIA CUDA decoding + NVENC encoding
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc -preset slow output.mp4

# Intel Quick Sync decoding + encoding
ffmpeg -hwaccel qsv -c:v h264_qsv -i input.mp4 -c:v h264_qsv output.mp4

# VA-API decoding + encoding
ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -i input.mp4 \
  -vf 'format=nv12,hwupload' -c:v h264_vaapi output.mp4
```

### Performance Options

```bash
# Multi-threading
ffmpeg -threads 4 -i input.mp4 output.mp4
ffmpeg -threads 0 -i input.mp4 output.mp4  # Auto detect

# Faster encoding (lower quality)
ffmpeg -i input.mp4 -preset ultrafast -crf 23 output.mp4

# Quality vs speed (presets)
# ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
ffmpeg -i input.mp4 -preset medium -crf 23 output.mp4

# Tune for specific content
ffmpeg -i input.mp4 -tune film output.mp4      # Film content
ffmpeg -i input.mp4 -tune animation output.mp4  # Animation
ffmpeg -i input.mp4 -tune grain output.mp4      # Grainy film
ffmpeg -i input.mp4 -tune stillimage output.mp4 # Slideshow
```

---

## Common Patterns

### Web-Optimized Video

```bash
# HTML5 video (MP4)
ffmpeg -i input.mp4 \
  -c:v libx264 -preset slow -crf 22 \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  -vf "scale=1280:720" \
  output.mp4

# WebM for web
ffmpeg -i input.mp4 \
  -c:v libvpx-vp9 -crf 30 -b:v 0 \
  -c:a libopus -b:a 128k \
  -vf "scale=1280:720" \
  output.webm

# Both formats for compatibility
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 22 -movflags +faststart video.mp4
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -c:a libopus video.webm
```

### Social Media Formats

```bash
# Instagram (1:1 square)
ffmpeg -i input.mp4 \
  -vf "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 -preset slow -crf 23 \
  -c:a aac -b:a 128k \
  instagram.mp4

# Instagram Stories (9:16)
ffmpeg -i input.mp4 \
  -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 -preset slow -crf 23 \
  -c:a aac -b:a 128k \
  story.mp4

# Twitter (16:9, < 512MB, < 2:20)
ffmpeg -i input.mp4 \
  -c:v libx264 -preset slow -crf 23 -maxrate 2M -bufsize 4M \
  -vf "scale=1280:720" \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  twitter.mp4

# YouTube (recommended settings)
ffmpeg -i input.mp4 \
  -c:v libx264 -preset slow -crf 18 \
  -c:a aac -b:a 192k \
  -vf "scale=1920:1080" \
  -r 30 \
  -movflags +faststart \
  youtube.mp4
```

### Batch Processing

```bash
# Convert all MP4 files to WebM
for f in *.mp4; do
  ffmpeg -i "$f" -c:v libvpx-vp9 -crf 30 "${f%.mp4}.webm"
done

# Batch resize
for f in *.mp4; do
  ffmpeg -i "$f" -vf "scale=1280:720" "resized_${f}"
done

# Batch extract audio
for f in *.mp4; do
  ffmpeg -i "$f" -vn -c:a libmp3lame -b:a 192k "${f%.mp4}.mp3"
done

# Parallel processing with GNU parallel
ls *.mp4 | parallel -j 4 ffmpeg -i {} -c:v libx264 -crf 23 {.}_converted.mp4
```

### Video from Images

```bash
# Create video from image sequence
ffmpeg -framerate 30 -pattern_type glob -i "*.jpg" -c:v libx264 -pix_fmt yuv420p output.mp4

# Specific pattern
ffmpeg -framerate 30 -i image_%04d.jpg -c:v libx264 output.mp4

# Slideshow with duration
ffmpeg -loop 1 -t 5 -i image.jpg -c:v libx264 -pix_fmt yuv420p output.mp4

# Slideshow from multiple images
ffmpeg -loop 1 -t 3 -i img1.jpg \
       -loop 1 -t 3 -i img2.jpg \
       -loop 1 -t 3 -i img3.jpg \
       -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0" \
       slideshow.mp4

# Ken Burns effect (zoom and pan)
ffmpeg -loop 1 -i image.jpg \
  -vf "zoompan=z='min(zoom+0.0015,1.5)':d=750:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080" \
  -c:v libx264 -t 30 output.mp4
```

### Screen Recording Conversion

```bash
# Optimize screen recording
ffmpeg -i screen_recording.mp4 \
  -c:v libx264 -preset slow -crf 18 \
  -vf "scale=1920:1080" \
  -c:a aac -b:a 128k \
  optimized.mp4

# Remove silence from screen recording
ffmpeg -i recording.mp4 \
  -af "silenceremove=1:0:-50dB" \
  no_silence.mp4
```

---

## Best Practices

### 1. Use Two-Pass Encoding for Best Quality

```bash
# Pass 1
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 1 -f mp4 /dev/null

# Pass 2
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 2 output.mp4
```

### 2. Use CRF for Variable Bitrate

```bash
# Better quality-to-size ratio
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4
```

### 3. Fast Start for Web Videos

```bash
# Move moov atom to beginning (faster streaming start)
ffmpeg -i input.mp4 -c copy -movflags +faststart output.mp4
```

### 4. Preserve Quality with Stream Copy

```bash
# When changing container only, use -c copy
ffmpeg -i input.mkv -c copy output.mp4
```

### 5. Use Proper Pixel Format

```bash
# Ensure compatibility (yuv420p for most players)
ffmpeg -i input.mp4 -pix_fmt yuv420p output.mp4
```

### 6. Optimize Presets

```bash
# Balance quality and encoding time
ffmpeg -i input.mp4 -preset slow -crf 22 output.mp4
```

### 7. Check Input First

```bash
# Always analyze before processing
ffprobe -show_streams input.mp4
```

### 8. Use Appropriate Audio Bitrate

```bash
# Don't waste space on audio
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -c:a aac -b:a 128k output.mp4
```

### 9. Batch Process Efficiently

```bash
# Use shell loops for multiple files
for f in *.mp4; do ffmpeg -i "$f" -c:v libx264 -crf 23 "${f%.mp4}_new.mp4"; done
```

### 10. Keep Original Aspect Ratio

```bash
# Use -1 to maintain aspect ratio
ffmpeg -i input.mp4 -vf "scale=1280:-1" output.mp4
```

---

## Troubleshooting

### Common Errors

```bash
# "Unknown encoder 'libx264'"
# Install ffmpeg with libx264 support
sudo apt install ffmpeg libx264-dev

# "Could not find codec parameters"
# File may be corrupted, try re-encoding
ffmpeg -err_detect ignore_err -i input.mp4 -c:v libx264 output.mp4

# "Invalid data found when processing input"
# Skip invalid data
ffmpeg -i input.mp4 -c copy -bsf:v h264_mp4toannexb output.mp4

# "Output file is empty"
# Check codecs and formats
ffprobe input.mp4
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4

# "Encoder did not produce proper pts"
# Add -vsync vfr
ffmpeg -i input.mp4 -vsync vfr output.mp4
```

### Audio/Video Sync Issues

```bash
# Fix A/V sync
ffmpeg -i input.mp4 -async 1 -vsync 1 output.mp4

# Delay audio by 2 seconds
ffmpeg -i input.mp4 -itsoffset 2 -i input.mp4 -map 0:v -map 1:a -c copy output.mp4

# Advance audio by 2 seconds
ffmpeg -i input.mp4 -itsoffset -2 -i input.mp4 -map 0:v -map 1:a -c copy output.mp4
```

### Quality Issues

```bash
# Improve quality (lower CRF)
ffmpeg -i input.mp4 -c:v libx264 -crf 18 output.mp4

# Two-pass for better quality
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 1 -f mp4 /dev/null
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 2 output.mp4

# Use better preset
ffmpeg -i input.mp4 -preset slower -crf 20 output.mp4
```

### Performance Issues

```bash
# Use hardware acceleration
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc output.mp4

# Use faster preset
ffmpeg -i input.mp4 -preset ultrafast output.mp4

# Limit CPU usage
ffmpeg -threads 2 -i input.mp4 output.mp4
```

### File Size Issues

```bash
# Reduce file size (increase CRF)
ffmpeg -i input.mp4 -c:v libx264 -crf 28 output.mp4

# Target specific file size (calculate bitrate)
# bitrate = (target_size_MB * 8192) / duration_seconds - audio_bitrate
ffmpeg -i input.mp4 -b:v 1000k -c:a aac -b:a 128k output.mp4

# Two-pass for exact size
ffmpeg -i input.mp4 -b:v 1000k -pass 1 -f mp4 /dev/null
ffmpeg -i input.mp4 -b:v 1000k -pass 2 output.mp4
```

---

## Quick Reference

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-i` | Input file | `-i input.mp4` |
| `-c:v` | Video codec | `-c:v libx264` |
| `-c:a` | Audio codec | `-c:a aac` |
| `-c copy` | Copy streams | `-c copy` |
| `-b:v` | Video bitrate | `-b:v 2M` |
| `-b:a` | Audio bitrate | `-b:a 128k` |
| `-crf` | Quality (lower=better) | `-crf 23` |
| `-preset` | Encoding speed | `-preset slow` |
| `-vf` | Video filter | `-vf "scale=1280:720"` |
| `-af` | Audio filter | `-af "volume=2.0"` |
| `-ss` | Start time | `-ss 00:01:30` |
| `-t` | Duration | `-t 00:00:10` |
| `-to` | End time | `-to 00:02:00` |
| `-r` | Frame rate | `-r 30` |
| `-s` | Resolution | `-s 1920x1080` |
| `-an` | No audio | `-an` |
| `-vn` | No video | `-vn` |
| `-sn` | No subtitles | `-sn` |
| `-map` | Stream selection | `-map 0:v:0` |
| `-y` | Overwrite output | `-y` |
| `-n` | Never overwrite | `-n` |

### Codec Shortcuts

| Codec | Video | Audio |
|-------|-------|-------|
| Copy | `-c:v copy` | `-c:a copy` |
| H.264 | `-c:v libx264` | - |
| H.265 | `-c:v libx265` | - |
| VP9 | `-c:v libvpx-vp9` | - |
| AV1 | `-c:v libaom-av1` | - |
| AAC | - | `-c:a aac` |
| MP3 | - | `-c:a libmp3lame` |
| Opus | - | `-c:a libopus` |
| Vorbis | - | `-c:a libvorbis` |

### Quality Presets

| Preset | Speed | Quality |
|--------|-------|---------|
| ultrafast | Fastest | Lowest |
| superfast | Very fast | Low |
| veryfast | Fast | Medium-low |
| faster | Moderate-fast | Medium |
| fast | Moderate | Good |
| medium | Moderate | Good (default) |
| slow | Slow | Very good |
| slower | Very slow | Excellent |
| veryslow | Slowest | Best |

### CRF Values

| Codec | Range | Default | Recommended |
|-------|-------|---------|-------------|
| H.264 | 0-51 | 23 | 18-28 |
| H.265 | 0-51 | 28 | 22-32 |
| VP9 | 0-63 | 30 | 15-35 |
| AV1 | 0-63 | 30 | 20-40 |

---

## Useful Resources

- **Official Documentation**: https://ffmpeg.org/documentation.html
- **Wiki**: https://trac.ffmpeg.org/wiki
- **Filters Documentation**: https://ffmpeg.org/ffmpeg-filters.html
- **Codecs**: https://ffmpeg.org/ffmpeg-codecs.html
- **Formats**: https://ffmpeg.org/ffmpeg-formats.html

---

FFmpeg is an incredibly powerful tool with nearly limitless capabilities for audio and video processing. Master these patterns and you'll be able to handle virtually any multimedia task from the command line.
