"""GPU temperature guard for CUDA training safety.

Provides a thermal guard that pauses training when GPU temperature exceeds
a configurable threshold and resumes once the temperature drops below a
recovery threshold. Supports NVML and nvidia-smi backends for temperature
reading, with automatic fallback.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import torch


class GPUTelemetryError(RuntimeError):
    """Fatal telemetry error indicating training must stop to avoid unsafe execution."""


@dataclass
class GPUTempCheckResult:
    """Result of a single GPU temperature check cycle.

    Attributes:
        temp_c: Current GPU temperature in degrees Celsius.
        paused: Whether training was paused during this check.
        pause_duration_s: Total seconds spent paused (0 if not paused).
        checks_during_pause: Number of temperature polls performed while paused.
        critical_seen: Whether the critical threshold was reached at any point.
    """

    temp_c: float
    paused: bool
    pause_duration_s: float
    checks_during_pause: int
    critical_seen: bool

    @property
    def repair_action(self) -> str:
        """Human-readable label describing the thermal action taken."""
        if not self.paused:
            return "none"
        prefix = "thermal_pause_critical" if self.critical_seen else "thermal_pause"
        return f"{prefix}_{int(round(self.pause_duration_s))}s"


class GPUTemperatureGuard:
    """Thermal guard with strict telemetry semantics for CUDA training.

    Monitors GPU temperature via NVML (primary) or nvidia-smi (fallback) and
    enforces pause/resume thresholds. Supports an optional critical threshold
    for triggering emergency offload actions.

    Attributes:
        is_active: Whether the guard is actively monitoring (CUDA available
            and guard enabled).
        pause_threshold_c: Temperature above which training is paused.
        resume_threshold_c: Temperature below which training resumes.
        critical_threshold_c: Optional critical temperature threshold.
        poll_interval_seconds: Seconds between temperature polls while paused.
        last_temperature_c: Most recently read temperature, or ``None``.
        total_pause_events: Cumulative count of pause events.
        total_paused_seconds: Cumulative seconds spent paused.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: str,
        nvml_device_index: int = 0,
        pause_threshold_c: float = 90.0,
        resume_threshold_c: float = 80.0,
        critical_threshold_c: Optional[float] = None,
        poll_interval_seconds: float = 30.0,
    ):
        """Initialize the GPU temperature guard.

        Args:
            enabled: Whether thermal guarding is enabled.
            device: Target device string (guard only activates for CUDA).
            nvml_device_index: NVML device index for multi-GPU systems.
            pause_threshold_c: Temperature in Celsius above which training
                is paused. Must be > 0.
            resume_threshold_c: Temperature in Celsius below which training
                resumes. Must be > 0 and < ``pause_threshold_c``.
            critical_threshold_c: Optional critical temperature. When set
                and reached, the caller may trigger emergency offload.
            poll_interval_seconds: Seconds between temperature polls while
                paused. Must be > 0.

        Raises:
            ValueError: If threshold or poll interval constraints are violated.
        """
        self._logger = logging.getLogger(__name__)
        self._device = str(device or "cpu")
        self._enabled = bool(enabled)
        self._nvml_device_index = int(nvml_device_index)
        self._pause_threshold_c = float(pause_threshold_c)
        self._resume_threshold_c = float(resume_threshold_c)
        self._critical_threshold_c = (
            None if critical_threshold_c is None else float(critical_threshold_c)
        )
        self._poll_interval_seconds = float(poll_interval_seconds)

        if self._pause_threshold_c <= 0:
            raise ValueError("pause_threshold_c must be > 0")
        if self._resume_threshold_c <= 0:
            raise ValueError("resume_threshold_c must be > 0")
        if self._resume_threshold_c >= self._pause_threshold_c:
            raise ValueError("resume_threshold_c must be < pause_threshold_c")
        if self._poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if self._critical_threshold_c is not None and self._critical_threshold_c <= 0:
            raise ValueError("critical_threshold_c must be > 0 when provided")

        self._active = (
            self._enabled
            and self._device.startswith("cuda")
            and torch.cuda.is_available()
        )

        self._nvml_module = None
        self._nvml_handle = None
        self._nvml_disabled = False
        self._nvml_fallback_logged = False

        self.last_temperature_c: Optional[float] = None
        self.total_pause_events = 0
        self.total_paused_seconds = 0.0

        if self._active:
            self._init_nvml()

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def pause_threshold_c(self) -> float:
        return self._pause_threshold_c

    @property
    def resume_threshold_c(self) -> float:
        return self._resume_threshold_c

    @property
    def critical_threshold_c(self) -> Optional[float]:
        return self._critical_threshold_c

    @property
    def poll_interval_seconds(self) -> float:
        return self._poll_interval_seconds

    def _init_nvml(self):
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._nvml_device_index)
            self._nvml_module = pynvml
        except Exception as exc:
            self._nvml_disabled = True
            if not self._nvml_fallback_logged:
                self._logger.warning(
                    "NVML unavailable for thermal guard, falling back to nvidia-smi: %s",
                    exc,
                )
                self._nvml_fallback_logged = True

    def _parse_temperature(self, raw: str) -> float:
        value = (raw or "").strip()
        if not value or value.lower() in {"n/a", "[not supported]"}:
            raise GPUTelemetryError(f"Invalid GPU temperature payload: {raw!r}")
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match is None:
            raise GPUTelemetryError(f"Unable to parse GPU temperature payload: {raw!r}")
        temp_c = float(match.group(0))
        if not math.isfinite(temp_c) or temp_c <= 0.0:
            raise GPUTelemetryError(f"Non-finite or invalid GPU temperature: {temp_c!r}")
        return temp_c

    def _read_temperature_from_nvml(self) -> float:
        if self._nvml_disabled or self._nvml_module is None or self._nvml_handle is None:
            raise GPUTelemetryError("NVML backend unavailable")
        try:
            pynvml = self._nvml_module
            return float(
                pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            )
        except Exception as exc:
            raise GPUTelemetryError(f"NVML temperature read failed: {exc}") from exc

    def _read_temperature_from_nvidia_smi(self) -> float:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            str(self._nvml_device_index),
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except FileNotFoundError as exc:
            raise GPUTelemetryError("nvidia-smi command not found") from exc
        except subprocess.CalledProcessError as exc:
            raise GPUTelemetryError(
                f"nvidia-smi failed (exit={exc.returncode}): {exc.stderr or exc.stdout}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GPUTelemetryError("nvidia-smi timed out") from exc
        except Exception as exc:
            raise GPUTelemetryError(f"Unexpected nvidia-smi failure: {exc}") from exc

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise GPUTelemetryError("nvidia-smi returned empty temperature output")
        return self._parse_temperature(lines[0])

    def read_temperature_c(self) -> float:
        """Read current GPU temperature in degrees Celsius.

        Attempts NVML first, falling back to nvidia-smi. Raises a fatal
        :class:`GPUTelemetryError` if both backends fail.

        Returns:
            Current GPU temperature in Celsius, or ``0.0`` if the guard
            is not active.

        Raises:
            GPUTelemetryError: If both NVML and nvidia-smi fail to provide
                a valid temperature reading.
        """
        if not self._active:
            self.last_temperature_c = None
            return 0.0

        nvml_error: Optional[Exception] = None
        if not self._nvml_disabled:
            try:
                temp_c = self._read_temperature_from_nvml()
                self.last_temperature_c = temp_c
                return temp_c
            except Exception as exc:
                nvml_error = exc
                self._nvml_disabled = True
                if not self._nvml_fallback_logged:
                    self._logger.warning(
                        "Thermal guard NVML read failed, falling back to nvidia-smi: %s",
                        exc,
                    )
                    self._nvml_fallback_logged = True

        try:
            temp_c = self._read_temperature_from_nvidia_smi()
            self.last_temperature_c = temp_c
            return temp_c
        except Exception as smi_exc:
            if nvml_error is None:
                raise GPUTelemetryError(f"GPU telemetry failed: {smi_exc}") from smi_exc
            raise GPUTelemetryError(
                f"GPU telemetry failed (NVML and nvidia-smi). "
                f"NVML: {nvml_error}; nvidia-smi: {smi_exc}"
            ) from smi_exc

    def wait_until_safe(self, *, context: str = "") -> GPUTempCheckResult:
        """Block until GPU temperature drops below the resume threshold.

        If the current temperature is already at or below the pause threshold,
        returns immediately without pausing. Otherwise, polls temperature at
        the configured interval until it drops below the resume threshold.

        Args:
            context: Optional human-readable label for log messages
                (e.g. ``"mlm epoch=1 batch=42"``).

        Returns:
            A :class:`GPUTempCheckResult` describing the temperature check
            outcome, including whether a pause occurred and its duration.
        """
        if not self._active:
            return GPUTempCheckResult(
                temp_c=0.0,
                paused=False,
                pause_duration_s=0.0,
                checks_during_pause=0,
                critical_seen=False,
            )

        temp_c = self.read_temperature_c()
        if temp_c <= self._pause_threshold_c:
            return GPUTempCheckResult(
                temp_c=temp_c,
                paused=False,
                pause_duration_s=0.0,
                checks_during_pause=0,
                critical_seen=(
                    self._critical_threshold_c is not None
                    and temp_c >= self._critical_threshold_c
                ),
            )

        critical_seen = (
            self._critical_threshold_c is not None
            and temp_c >= self._critical_threshold_c
        )
        checks = 0
        started = time.perf_counter()
        context_label = context or "training"

        self._logger.warning(
            "[ThermalGuard] %s paused: GPU %.1fC > %.1fC (resume <= %.1fC)",
            context_label,
            temp_c,
            self._pause_threshold_c,
            self._resume_threshold_c,
        )
        if critical_seen:
            self._logger.warning(
                "[ThermalGuard] %s critical threshold reached: %.1fC >= %.1fC",
                context_label,
                temp_c,
                float(self._critical_threshold_c),
            )

        while temp_c > self._resume_threshold_c:
            time.sleep(self._poll_interval_seconds)
            checks += 1
            temp_c = self.read_temperature_c()
            if (
                self._critical_threshold_c is not None
                and temp_c >= self._critical_threshold_c
            ):
                critical_seen = True
            self._logger.warning(
                "[ThermalGuard] %s waiting: GPU %.1fC (resume <= %.1fC)",
                context_label,
                temp_c,
                self._resume_threshold_c,
            )

        duration_s = time.perf_counter() - started
        self.total_pause_events += 1
        self.total_paused_seconds += duration_s
        self._logger.warning(
            "[ThermalGuard] %s resumed after %.1fs at %.1fC",
            context_label,
            duration_s,
            temp_c,
        )

        return GPUTempCheckResult(
            temp_c=temp_c,
            paused=True,
            pause_duration_s=duration_s,
            checks_during_pause=checks,
            critical_seen=critical_seen,
        )
