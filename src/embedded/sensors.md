# Sensors & Sensor Fusion

## Overview

Reading a sensor is rarely just "read the register" — between the physical quantity and a number your application can trust sits a whole **signal chain**: the sensor's analog front-end, sampling, calibration, filtering, and (when several sensors measure overlapping quantities) **fusion** that combines them into one better estimate than any single sensor gives. This page covers that chain on an MCU: how sensors connect ([ADC](adc.md), [I2C](i2c.md), [SPI](spi.md), [1-Wire](one_wire.md)), how to sample without aliasing, how to calibrate, and the fusion algorithms — complementary and Kalman filters — that turn a noisy [IMU](#imus-and-the-fusion-problem) into a stable orientation. The math here leans on [DSP](dsp.md) techniques.

```
   Physical    ┌─────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  Application
   quantity ──▶│ Sensor  │─▶│ Sample │─▶│ Calibrate│─▶│ Filter │─▶ usable value
   (temp,      │ + AFE   │  │ (ADC/  │  │ offset/  │  │ (LPF/  │   + fused
    motion)    │         │  │  bus)  │  │ scale)   │  │ fusion)│     estimate
              └─────────┘  └────────┘  └──────────┘  └────────┘
```

## How Sensors Connect

| Interface | Typical sensors | Notes |
|-----------|-----------------|-------|
| **[ADC](adc.md)** (analog) | Thermistors, photodiodes, load cells, potentiometers | Raw voltage; you do the conversion math |
| **[I2C](i2c.md)** | IMUs, pressure, ambient light, RTCs | Cheap, multi-drop, register-based |
| **[SPI](spi.md)** | High-rate IMUs, ADCs, pressure | Faster, for high sample rates |
| **[1-Wire](one_wire.md)** | DS18B20 temperature | One wire, long cheap cable |
| **Digital pulse** | Hall, flow, anemometer | Count edges with a [timer](timers.md) |
| **PDM/[I2S](i2s.md)** | MEMS microphones | Needs decimation ([DSP](dsp.md)) |

Digital sensors give you calibrated-ish engineering units; analog sensors hand you a raw voltage and leave the entire conversion, linearization, and calibration to you.

## Sampling Without Lying to Yourself

### Nyquist and aliasing

To represent a signal with frequency content up to `f`, you must sample at **>2f**. Sample too slowly and high-frequency content **aliases** — folds down and masquerades as a low-frequency signal you can never remove in software:

```
  Real 90 Hz vibration, sampled at 100 Hz → aliases to a fake 10 Hz wobble
  No digital filter can recover the truth; the damage is done at the ADC.
```

The fix is an **analog anti-alias filter** (a simple RC low-pass) *before* the [ADC](adc.md), plus an adequate sample rate. This is a hardware decision, not a software one.

### Oversampling & decimation

Sampling faster than needed and averaging trades speed for resolution: averaging `4^n` samples gains ~`n` bits of effective resolution (for uncorrelated noise) and also pushes the anti-alias requirement up. A cheap way to get a 14-bit reading from a 12-bit [ADC](adc.md).

## Calibration

Every real sensor has **offset** (reads non-zero at zero input) and **scale/gain** error (slope wrong). The minimal model:

```
   value = (raw - offset) × scale
```

- **Two-point calibration** measures `raw` at two known references to solve offset and scale.
- **Temperature drift:** many sensors' offset/scale change with temperature — high-accuracy systems store a per-temperature correction table (often in [EEPROM/flash](flash_filesystems.md), written at factory test).
- **IMU-specific:** accelerometer needs 6-position (±1 g per axis) calibration; magnetometer needs **hard-iron** (offset) and **soft-iron** (scale/skew) calibration by rotating through all orientations and fitting an ellipsoid to a sphere.

## Filtering

Most sensor data needs smoothing without destroying the signal you care about:

| Filter | Use | Cost |
|--------|-----|------|
| **Moving average** | Simple smoothing | Cheap, but poor frequency response |
| **Exponential (IIR LPF)** | `y += α(x − y)` — one line, tunable | Very cheap, the workhorse |
| **Median** | Reject spikes/outliers | Good for impulse noise |
| **FIR** | Sharp, linear-phase | More compute; [CMSIS-DSP](dsp.md) |

The one-line exponential low-pass `y += alpha * (x - y)` is the embedded default — `alpha` near 1 = responsive/noisy, near 0 = smooth/laggy. See [DSP](dsp.md) for the filter theory.

## IMUs and the Fusion Problem

An IMU combines an **accelerometer** (measures gravity + linear acceleration), a **gyroscope** (measures rotation rate), and often a **magnetometer** (measures heading). Each alone is inadequate for orientation:

```
  Gyro:   smooth, fast, accurate short-term — but DRIFTS (integrating rate
          accumulates error → heading wanders over seconds/minutes)
  Accel:  absolute tilt reference (gravity) — but NOISY and corrupted by
          any linear motion/vibration
  Mag:    absolute heading — but corrupted by nearby metal/currents

  Fusion: trust gyro short-term, correct its drift with accel/mag long-term
```

### Complementary filter

The pragmatic 80% solution — a high-pass on the gyro plus a low-pass on the accelerometer, blended:

```c
// pitch from gyro (fast) corrected by accel (absolute), per loop dt
angle = (1 - a) * (angle + gyro_rate * dt)   // integrate gyro, high-pass
      +      a  * accel_angle;                // accel tilt, low-pass
// a ≈ 0.02 : mostly gyro, slowly pulled toward accel truth
```

Cheap, stable, and good enough for self-balancing robots, camera gimbals, and drones at the hobby level.

### Kalman filter

The statistically optimal approach: maintain a state estimate *and its uncertainty*, predict it forward with the gyro, then correct with the accel/mag weighted by their relative confidence. **Predict-update** each cycle:

```
   Predict:  state ← model(state, gyro);  uncertainty grows
   Update:   compare to accel/mag measurement;
             Kalman gain weights model vs measurement by their variances;
             state corrected, uncertainty shrinks
```

It's heavier (matrix math — [CMSIS-DSP](dsp.md) or fixed-point) and needs tuning of process/measurement noise. For attitude specifically, the **Madgwick** and **Mahony** filters are popular lighter-weight quaternion alternatives that get near-Kalman quality at complementary-filter cost.

## Where this connects

- [ADC](adc.md) — analog sensor front-end; oversampling and anti-aliasing live here.
- [I2C](i2c.md) / [SPI](spi.md) / [1-Wire](one_wire.md) — digital sensor buses.
- [DSP & Fixed-Point](dsp.md) — the filter math (FIR/IIR, fixed-point) and PDM decimation.
- [Timers](timers.md) — fixed-rate sampling triggers and pulse/frequency sensors.
- [DMA](dma.md) — streaming high-rate sensor samples without per-sample CPU load.
- [Motor Control](motor_control.md) — encoders/resolvers as position sensors in the control loop.
- [Flash Filesystems](flash_filesystems.md) — storing per-unit calibration constants.

## Pitfalls

1. **Aliasing from under-sampling.** Once high-frequency content folds into your band, no software filter recovers it. Add an analog anti-alias filter and sample fast enough.
2. **Integrating gyro alone for angle.** Drift makes the estimate wander within seconds. Always fuse with an absolute reference (accel/mag).
3. **Skipping magnetometer calibration.** Uncalibrated hard/soft-iron errors make heading useless near any metal or current-carrying wire.
4. **Filtering away the signal.** Too-aggressive smoothing adds lag that destabilizes control loops. Match filter bandwidth to the real dynamics.
5. **Trusting accel during motion.** Accelerometer "tilt" is wrong whenever the device accelerates/vibrates; weight it low and reject during high linear acceleration.
6. **Ignoring temperature drift.** Offset/scale move with temperature; precision applications need a temperature-compensation table.
7. **Reading a sensor before it's ready.** Many sensors have conversion/settling times or a data-ready line; reading early returns stale or default values.
8. **Float math at high rates without an FPU.** A Kalman filter at kHz rates needs an FPU or fixed-point ([CMSIS-DSP](dsp.md)).

## See Also

- [ADC](adc.md) — analog front-end and oversampling
- [DSP & Fixed-Point Math](dsp.md) — filter theory and implementation
- [I2C](i2c.md) / [SPI](spi.md) — digital sensor buses
- [Motor Control](motor_control.md) — position feedback sensors
