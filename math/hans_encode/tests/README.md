# HANS Ascend 950 validation

Run the commands below from the repository root on Linux with a CANN
toolchain and CPU-debug libraries that support Ascend 950. Record the exact
CANN build used; this guide does not specify a minimum or first release.

## Host tiling UT

```sh
bash build.sh -u --ophost --soc=ascend950 --ops=hans_encode,hans_decode --gtest_filter='*HansEncode950*:*HansDecode950*'
```

The filter selects 40 cases (28 encode, 12 decode), including parameterized
cases. They exercise registered tiling functions with explicit platform
resources. Coverage includes FP16/BF16/FP32, statistic/reshuff attributes,
buffer and workspace sizes, core limits, partial tiles, invalid parameters,
and the existing Ascend910B capacity branch. Omit the filter to include the
other HANS host tests.

## Kernel UT (CPU debug)

```sh
# Encode only
bash build.sh -u --opkernel --soc=ascend950 --ops=hans_encode --gtest_filter='HansEncode950KernelTest.*'

# Decode only
bash build.sh -u --opkernel --soc=ascend950 --ops=hans_decode --gtest_filter='HansDecode950KernelTest.*'

# Both suites
bash build.sh -u --opkernel --soc=ascend950 --ops=hans_encode,hans_decode --gtest_filter='HansEncode950KernelTest.*:HansDecode950KernelTest.*'
```

The targets compile the 950 apt entries and run through `ICPU_RUN_KF`.
Running these commands on an NPU host still uses CPU debug, not the NPU.

- Encode registers 39 cases; 23 currently return `GTEST_SKIP()` because of
  unresolved CPU-debug failures. The other 16 exercise FP32 encoding,
  var-only output and boundary checks. Active encode work uses one core;
  names containing `multicore` do not establish multicore coverage.
- Decode registers 44 cases. Checks include exact output bytes for var-only
  and constructed single-tile fixed streams, both byte-width
  specializations, reshuffling, multiple cores and invalid metadata.
- Kernel fixtures may bypass host validation to test empty/small inputs and
  insufficient capacity. They do not extend the supported operator inputs.
  Invalid-stream checks do not guarantee rollback after partial output.

For commit `113ef4201`, [CI #3433](https://gitcode.com/cann/ops-math/actions/runs/2543cd175153448d9ac04698bdb9ba60)
reported 60 passed and 23 skipped: encode 16 passed/23 skipped, decode
44 passed. This is not an 83-case pass or an NPU round-trip result.
Revalidate after source or environment changes; do not count skips as passes.

## Ascend 950 output capacity

Let N be the input element count (a multiple of 64, at least 32768), and B
the element width in bytes. The mantissa buffer must contain N * (B - 1)
bytes. Define:

```text
C = min(floor(N / 32768), available AIV cores, 56)
V = 64 * (floor((N / 64) / C) + (N / 64) % C)
fixed-only bound (bytes) = 512 + C * (V + ceil(V / 4096) * 128 + 8448)
```

C must be positive; V is the longest core's element count. The bound includes
the common header, per-tile metadata and state tails. It is independent of
PDF contents and may exceed the space needed for a particular input.

- With reshuff enabled, fixed must meet this bound, which is also used for
  fixed staging in workspace.
- Without reshuff, fixed must be at least 512 bytes. Tiling checks the var
  capacity against the remaining symbols after the guaranteed fixed prefix
  of each core; total fixed + var capacity alone is insufficient.
  A 512-byte fixed buffer requires N var bytes.
- The implementation also checks signed header-field limits. These rules
  are specific to Ascend 950; other platforms retain their existing rules.

## NPU validation

Build the production kernel separately:

```sh
bash build.sh --opkernel --soc=ascend950 --ops=hans_encode,hans_decode
```

A successful build is not a device test. Use the built operator package for
encode/decode round trips on Ascend 950 and compare the recovered bytes
against the original input, including NaN payloads and signed zero.

Cover FP16/BF16/FP32, supplied/recomputed PDF, both reshuffle modes,
single/multiple cores, fixed-only/var-only/mixed outputs, partial tiles and
multi-tile overflow. Decode CPU fixtures currently lack mixed fixed/var and
multi-tile overflow coverage. Check malformed streams separately and measure
performance on the NPU, not in CPU debug.

Attach the source commit, exact CANN build, driver/firmware, device model,
commands and complete logs. For round trips, also record dtype, shape,
PDF mode, buffer capacities, reshuffle mode and bytewise comparison results.
