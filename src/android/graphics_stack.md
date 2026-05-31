# Graphics Stack

## Overview

Android's graphics stack turns what apps draw into pixels on screen, smoothly and in sync with
the display's refresh rate. The key players are **Surfaces** and **BufferQueues** (the
producer/consumer buffer plumbing), **SurfaceFlinger** (the system compositor),
**Hardware Composer (HWC)** (offloads compositing to display hardware), **Gralloc** (graphics
buffer allocator), and **Choreographer/VSYNC** (timing). Understanding this pipeline is the key
to diagnosing **jank** (dropped frames).

See [SystemServer & Core Services](system_server.md) for WindowManager's role and
[ART & Dalvik Runtime](art_runtime.md) for GC pauses that can cause jank.

## The Producer/Consumer Model

The whole stack is built on **BufferQueue**: a queue of graphics buffers shared between a
**producer** (something that renders, e.g. an app, the camera, a video decoder) and a
**consumer** (something that displays/composites, e.g. SurfaceFlinger).

```text
Producer (app render)            BufferQueue            Consumer (SurfaceFlinger)
   dequeueBuffer() ──▶  [ free | filling | queued ] ──▶  acquireBuffer()
   draw into buffer                                       composite buffer
   queueBuffer()   ──────────────────────────────────▶   releaseBuffer() (back to free)
```

- A **Surface** is the producer interface to a BufferQueue (`android.view.Surface`).
- **SurfaceFlinger** is the consumer for on-screen surfaces.
- Buffers are allocated by **Gralloc** and are typically **GPU/display-shareable** (often
  backed by `dma-buf`), so they can be passed by handle without copying.

## Surfaces, SurfaceFlinger & WindowManager

Each window has one or more surfaces. **WindowManagerService** (in
[system_server](system_server.md)) decides window geometry, z-order, and visibility, and hands
that layer metadata to **SurfaceFlinger**. SurfaceFlinger takes the latest buffer from each
visible surface and **composites** them into the final screen image.

```text
App windows / system UI / wallpaper          (each a layer with its own BufferQueue)
        │ buffers + layer state (from WindowManager)
        ▼
   SurfaceFlinger  ──decides──▶  HWC (overlay) and/or GPU (GL/Vulkan) composition
        │
        ▼
   Display (framebuffer / DPU)
```

### Hardware Composer (HWC)

Instead of using the GPU to blend every layer, SurfaceFlinger asks the **HWC HAL** which layers
the display hardware can compose directly (overlays). Offloading to HWC saves GPU and power.
Layers HWC can't handle are composited by the GPU into a single buffer, then handed back to HWC.

```bash
adb shell dumpsys SurfaceFlinger             # layers, composition strategy, refresh rate
adb shell dumpsys SurfaceFlinger --latency <LayerName>   # frame latency stats
adb shell service call SurfaceFlinger 1013   # current frame number (debug)
```

### Gralloc

**Gralloc** ("graphics alloc") is the HAL that allocates and maps graphics buffers with the
right format, usage flags (GPU render, texture, composer, video), and memory backing so all
producers/consumers (GPU, display, codec, camera) can share them.

## VSYNC & Choreographer

**VSYNC** is the periodic signal marking when the display starts a new refresh (e.g. every
16.6 ms at 60 Hz, 8.3 ms at 120 Hz). Android aligns rendering and composition to VSYNC to avoid
tearing and to pace frames.

**Choreographer** (app side) registers a callback on each VSYNC to drive input → animation →
draw, so the app produces one frame per display refresh.

```text
VSYNC ─┬───────────────┬───────────────┬──────────────▶ time (one tick per refresh)
       │ app: input    │ app: input    │
       │ + animation   │ + animation   │
       │ + draw (RT)   │ + draw (RT)   │
       │ SF: composite │ SF: composite │
       ▼               ▼               ▼
     present         present         present
```

Modern Android uses **VSYNC offsets** and a render pipeline (UI thread → **RenderThread** →
GPU) so app rendering, SurfaceFlinger composition, and scan-out are staggered to maximize the
time budget for each stage.

```kotlin
// App-side frame callback (rarely needed directly; the view system uses it internally)
Choreographer.getInstance().postFrameCallback { frameTimeNanos ->
    // do per-frame work aligned to VSYNC
}
```

### Variable refresh rate

Android supports multiple/variable refresh rates; SurfaceFlinger and the display pick a rate
based on content (e.g. drop to 60 Hz for static UI, 120 Hz for scrolling) to save power.

## The Frame Lifecycle (end-to-end)

```text
1. Input event delivered (InputManager → app via VSYNC-aligned Choreographer)
2. UI thread: measure/layout/draw → records a display list
3. RenderThread: issues GPU commands (Skia → GL/Vulkan) into a buffer (dequeued from BufferQueue)
4. queueBuffer() → buffer enters SurfaceFlinger's queue
5. On VSYNC, SurfaceFlinger acquires latest buffers, composites (HWC + GPU), presents
6. Display scans out the composited frame
```

If any stage misses its VSYNC deadline, a frame is dropped → **jank**. Common causes: heavy
work on the UI thread, allocations triggering [GC pauses](art_runtime.md), overdraw, large
bitmaps, or synchronous I/O on the main thread.

## Diagnosing Jank

```bash
# Per-frame timing histogram for an app (look at "Janky frames")
adb shell dumpsys gfxinfo com.example.app

# Capture a system trace (open in ui.perfetto.dev) — see UI thread, RenderThread, SurfaceFlinger
adb shell perfetto -o /data/misc/perfetto-traces/trace -t 10s sched gfx view
```

- **Android Studio Profiler / Perfetto**: visualize UI thread vs RenderThread vs SurfaceFlinger.
- **GPU overdraw** debug (Developer Options → Debug GPU overdraw).
- **Profile HWUI rendering** bars to spot which stage exceeds the frame budget.
- **Jetpack Macrobenchmark** `FrameTimingMetric` for automated jank measurement.

## Graphics APIs (app/native)

| API | Layer | Use |
|-----|-------|-----|
| Canvas / HWUI (Skia) | Framework | Standard View/Compose drawing (GPU-accelerated) |
| OpenGL ES | Native/Java | Custom 2D/3D rendering |
| Vulkan | Native | Low-overhead, explicit GPU control; modern games/engines |
| `SurfaceView` / `TextureView` | Framework | Dedicated surfaces for video/camera/GL |
| EGL | Native | Binds GL/Vulkan contexts to surfaces |

## Best Practices

1. **Keep the UI thread free** — move work off it; never do I/O or heavy parsing in `onDraw`.
2. **Avoid allocations per frame** to prevent [GC](art_runtime.md)-induced jank.
3. **Reduce overdraw** — flatten layouts, avoid stacked opaque backgrounds.
4. **Right-size bitmaps** to display dimensions; reuse/pool large buffers.
5. **Use `SurfaceView` for high-rate content** (video/camera/games) so it composites independently.
6. **Measure with `gfxinfo` / Perfetto / Macrobenchmark**, not by eye.
7. **Respect VSYNC** — let the framework pace frames; don't busy-render.

## Resources

- [Graphics architecture — AOSP](https://source.android.com/docs/core/graphics/architecture)
- [BufferQueue & Gralloc](https://source.android.com/docs/core/graphics/arch-bq-gralloc)
- [SurfaceFlinger & HWC](https://source.android.com/docs/core/graphics/arch-sf-hwc)
- [VSYNC](https://source.android.com/docs/core/graphics/implement-vsync)
- [Slow rendering / jank — Android Developers](https://developer.android.com/topic/performance/vitals/render)

### Related Files

- [SystemServer & Core Services](system_server.md) — WindowManager feeds SurfaceFlinger
- [ART & Dalvik Runtime](art_runtime.md) — GC pauses as a jank source
- [Jetpack](jetpack.md) — Compose rendering builds on this stack
