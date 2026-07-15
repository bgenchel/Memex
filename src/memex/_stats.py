"""Statistics returned by :class:`memex.MemoryExecutor`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceStats:
    device: str
    total_mib: int
    available_mib: int
    reserved_mib: int
    running: int


@dataclass(frozen=True)
class ExecutorStats:
    backend: str
    running: int
    queued: int
    retrying: int
    completed: int
    failed: int
    cancelled: int
    observed_peak_mib: int | None
    headroom_mib: int
    devices: tuple[DeviceStats, ...]

    def __str__(self) -> str:
        peak = "unknown" if self.observed_peak_mib is None else f"{self.observed_peak_mib} MiB"
        lines = [
            "MemoryExecutor",
            f"Backend: {self.backend.upper()}",
            "Tasks",
            f"  Running:   {self.running}",
            f"  Queued:    {self.queued}",
            f"  Retrying:  {self.retrying}",
            f"  Completed: {self.completed}",
            f"  Failed:    {self.failed}",
            f"  Cancelled: {self.cancelled}",
            "Learned memory",
            f"  Peak observed: {peak}",
            f"  Headroom:      {self.headroom_mib} MiB",
            "Devices",
        ]
        for device in self.devices:
            lines.extend(
                [
                    f"  {device.device}",
                    f"    Total:       {device.total_mib} MiB",
                    f"    Available:   {device.available_mib} MiB",
                    f"    Reserved:    {device.reserved_mib} MiB",
                    f"    Running:     {device.running}",
                ]
            )
        return "\n".join(lines)

