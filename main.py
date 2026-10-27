"""A small 12-TET vs. fixed C-major just intonation listening demo."""

from __future__ import annotations

import math
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 48_000
BLOCK_SIZE = 512
ATTACK_RELEASE_SECONDS = 0.035
A4 = 440.0
# This is the requested common C reference.  The 12-TET A4=440 calculation
# produces 261.6255653006 Hz, which is the same value at the shown precision.
C4 = 261.625565

MIDI_NOTES = {
    "C": 60, "D": 62, "E": 64, "F": 65,
    "G": 67, "A": 69, "B": 71,
}
JI_RATIOS = {
    "C": 1 / 1, "D": 9 / 8, "E": 5 / 4, "F": 4 / 3,
    "G": 3 / 2, "A": 5 / 3, "B": 15 / 8,
}
CHORDS = {
    "C major": (("C", 4), ("E", 4), ("G", 4)),
    "A minor": (("A", 3), ("C", 4), ("E", 4)),
    "F major": (("F", 3), ("A", 3), ("C", 4)),
    "G major": (("G", 3), ("B", 3), ("D", 4)),
}


def equal_temperament_frequency(note: str, octave: int) -> float:
    """Return a 12-TET frequency using the A4 = 440 Hz MIDI formula."""
    midi_note = MIDI_NOTES[note] + 12 * (octave - 4)
    return A4 * 2 ** ((midi_note - 69) / 12)


def just_intonation_frequency(note: str, octave: int) -> float:
    """Return a fixed C-major JI frequency; it never depends on the chord."""
    return C4 * JI_RATIOS[note] * 2 ** (octave - 4)


@dataclass
class Voice:
    frequencies: np.ndarray
    phases: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    level: float = 0.0
    target: float = 1.0


class Synthesizer:
    """One output stream mixing all chord voices in its callback thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._voices: dict[tuple[str, str], Voice] = {}
        self._master_volume = 0.55
        self._overtones = True
        self._stream: sd.OutputStream | None = None
        self._level_step = 1.0 / (ATTACK_RELEASE_SECONDS * SAMPLE_RATE)

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_chord(self, tuning: str, chord: str, enabled: bool) -> None:
        key = (tuning, chord)
        with self._lock:
            if enabled:
                frequencies = self._chord_frequencies(tuning, chord)
                voice = self._voices.get(key)
                if voice is None:
                    self._voices[key] = Voice(frequencies=frequencies)
                else:
                    # A release can be cancelled without resetting its phases.
                    voice.target = 1.0
            elif key in self._voices:
                self._voices[key].target = 0.0

    def stop_all(self) -> None:
        with self._lock:
            for voice in self._voices.values():
                voice.target = 0.0

    def set_master_volume(self, value: float) -> None:
        with self._lock:
            self._master_volume = max(0.0, min(1.0, value))

    def set_overtones(self, enabled: bool) -> None:
        with self._lock:
            self._overtones = enabled

    @staticmethod
    def _chord_frequencies(tuning: str, chord: str) -> np.ndarray:
        frequency = (equal_temperament_frequency if tuning == "12-TET"
                     else just_intonation_frequency)
        return np.array([frequency(note, octave) for note, octave in CHORDS[chord]],
                        dtype=np.float64)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        # Locking only protects tiny shared state changes from the GUI thread.
        with self._lock:
            mix = np.zeros(frames, dtype=np.float64)
            completed: list[tuple[str, str]] = []
            sample_numbers = np.arange(frames, dtype=np.float64)
            harmonic_numbers = ((1, 0.55, 0.28, 0.14) if self._overtones else (1,))

            for key, voice in self._voices.items():
                phase_steps = (2 * math.pi * voice.frequencies[:, None]
                               * sample_numbers / SAMPLE_RATE)
                phases = voice.phases[:, None] + phase_steps
                waveform = np.zeros(frames, dtype=np.float64)
                for harmonic, amplitude in enumerate(harmonic_numbers, start=1):
                    waveform += amplitude * np.sin(phases * harmonic).sum(axis=0)
                waveform /= (len(voice.frequencies) * sum(harmonic_numbers))

                previous = voice.level
                levels = np.clip(previous + self._level_step * voice.target * (sample_numbers + 1)
                                 - self._level_step * (1 - voice.target) * (sample_numbers + 1),
                                 0.0, 1.0)
                voice.level = float(levels[-1])
                voice.phases = (voice.phases + 2 * math.pi * voice.frequencies * frames / SAMPLE_RATE) % (2 * math.pi)
                mix += waveform * levels
                if voice.target == 0.0 and voice.level <= 0.0:
                    completed.append(key)

            for key in completed:
                del self._voices[key]

            # A soft limiter prevents overload without changing relative tuning.
            mix = np.tanh(mix * self._master_volume * 1.35)
            outdata[:, 0] = mix.astype(np.float32)


class TuningDemoApp:
    ACTIVE_COLOR = "#257a45"
    INACTIVE_COLOR = "#e8e8e8"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Just Intonation vs. Equal Temperament")
        self.root.resizable(False, False)
        self.synth = Synthesizer()
        self.active: dict[tuple[str, str], bool] = {
            (tuning, chord): False for tuning in ("12-TET", "Just Intonation") for chord in CHORDS
        }
        self.buttons: dict[tuple[str, str], tk.Button] = {}
        self.status_var = tk.StringVar(value="0 active chords")
        self.overtones_var = tk.BooleanVar(value=True)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            self.synth.start()
        except Exception as exc:
            messagebox.showerror("Audio device unavailable", f"Could not start audio output:\n{exc}")
            self.root.after(10, self.root.destroy)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(sticky="nsew")
        ttk.Label(outer, text="12-TET and fixed C-major just intonation", font=("TkDefaultFont", 14, "bold")).grid(
            row=0, column=0, columnspan=5, sticky="w", pady=(0, 4))
        ttk.Label(outer, text="Toggle any chords together to hear their tuning differences.").grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(0, 12))

        for column, chord in enumerate(CHORDS, start=1):
            notes = "–".join(f"{note}{octave}" for note, octave in CHORDS[chord])
            ttk.Label(outer, text=f"{chord}\n{notes}", justify="center").grid(row=2, column=column, padx=4, pady=(0, 6))

        for row, tuning in enumerate(("12-TET", "Just Intonation"), start=3):
            label = "Equal Temperament\n(12-TET)" if tuning == "12-TET" else "Just Intonation\n(fixed C major)"
            ttk.Label(outer, text=label, justify="right").grid(row=row, column=0, sticky="e", padx=(0, 8))
            for column, chord in enumerate(CHORDS, start=1):
                button = tk.Button(outer, text="Off", width=14, height=2,
                                   bg=self.INACTIVE_COLOR, activebackground="#d5d5d5",
                                   command=lambda t=tuning, c=chord: self._toggle(t, c))
                button.grid(row=row, column=column, padx=4, pady=4)
                self.buttons[(tuning, chord)] = button

        controls = ttk.LabelFrame(outer, text="Controls", padding=10)
        controls.grid(row=5, column=0, columnspan=5, sticky="ew", pady=(14, 0))
        ttk.Label(controls, text="Master volume").grid(row=0, column=0, sticky="w")
        volume = ttk.Scale(controls, from_=0.0, to=1.0, value=0.55,
                           command=lambda value: self.synth.set_master_volume(float(value)))
        volume.grid(row=0, column=1, padx=8, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Checkbutton(controls, text="Organ-like overtones", variable=self.overtones_var,
                        command=lambda: self.synth.set_overtones(self.overtones_var.get())).grid(row=0, column=2, padx=8)
        ttk.Button(controls, text="Stop All", command=self._stop_all).grid(row=0, column=3, padx=(8, 0))
        ttk.Label(outer, textvariable=self.status_var).grid(row=6, column=0, columnspan=5, pady=(10, 0))

    def _toggle(self, tuning: str, chord: str) -> None:
        key = (tuning, chord)
        enabled = not self.active[key]
        self.active[key] = enabled
        self.synth.set_chord(tuning, chord, enabled)
        button = self.buttons[key]
        button.configure(text="On" if enabled else "Off",
                         bg=self.ACTIVE_COLOR if enabled else self.INACTIVE_COLOR,
                         fg="white" if enabled else "black",
                         activebackground="#1d6137" if enabled else "#d5d5d5")
        self._update_status()

    def _stop_all(self) -> None:
        self.synth.stop_all()
        for key, button in self.buttons.items():
            self.active[key] = False
            button.configure(text="Off", bg=self.INACTIVE_COLOR, fg="black", activebackground="#d5d5d5")
        self._update_status()

    def _update_status(self) -> None:
        count = sum(self.active.values())
        self.status_var.set(f"{count} active chord{'s' if count != 1 else ''}")

    def _on_close(self) -> None:
        self.synth.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    TuningDemoApp(root)
    root.mainloop()
