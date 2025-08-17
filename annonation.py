import gc
import gzip
import importlib
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sphn
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper
from scipy.io.wavfile import write as write_wav

# Ensure the correct whisper_timestamped transcribe module is available
transcribe = importlib.import_module("whisper_timestamped.transcribe")
old_get_vad_segments = transcribe.get_vad_segments
logger = logging.getLogger(__name__)

# --- Helper Functions and Classes (from original script) ---

SAMPLE_RATE = 16_000

@contextmanager
def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp", pid=False):
    """
    Writes to a temporary file and then renames it, preventing corrupted
    files if the script is interrupted.
    """
    tmp_path = str(path) + suffix
    if pid:
        tmp_path += f".{os.getpid()}"
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)


def load_audio_paths(egs_path: Path) -> list[Path]:
    """
    Loads audio paths from a JSONL egs file (can be gzipped).
    An EGS file is a manifest, where each line is a JSON object
    containing metadata about an audio file, including its path.
    Example line: {"path": "/path/to/your/audio.wav", "duration": 10.5}
    """
    open_fn = gzip.open if str(egs_path).lower().endswith(".gz") else open
    with open_fn(egs_path, "rt", encoding="utf-8") as fp:
        lines = fp.readlines()
    paths: list[Path] = []
    for line in lines:
        d = json.loads(line)
        paths.append(Path(d["path"]))
    return paths


def init_logging(verbose: bool = False):
    """Initializes logging to stream to stderr."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


@dataclass
class Params:
    """Configuration parameters for the transcription process."""
    egs: Path
    verbose: bool
    lang: str
    whisper_model: str
    keep_silence_in_segments: float
    rerun_errors: bool
    shards: int = 1
    shard: int = 0


def process_one(
    in_file: Path,
    out_file: Path,
    language: str,
    w_model,
    params: "Params",
    channel: int = 0,
):
    """
    Processes a single audio file: loads, transcribes with Whisper,
    and saves the timestamped words to a JSON file.
    """
    logger.debug("Loading audio %s", in_file)
    gc.collect()
    torch.cuda.empty_cache()

    try:
        x, sr = sphn.read(in_file)
    except Exception as e:
        logger.error(f"Could not read audio file {in_file}: {e}")
        return

    x = torch.from_numpy(x).cuda()
    dur = x.shape[-1] / sr
    if dur > 3600 * 4:
        logger.warning(f"File {in_file} is longer than 4 hours and will be skipped.")
        return

    vocals = x[channel][None]
    vocals = F.resample(vocals, sr, SAMPLE_RATE)
    sr = SAMPLE_RATE

    def new_get_vad_segments(*args, **kwargs):
        """A wrapper to add padding around VAD segments."""
        segs = old_get_vad_segments(*args, **kwargs)
        outs = []
        last_end = 0
        d = int(SAMPLE_RATE * params.keep_silence_in_segments)
        logger.debug("Reintroducing %d samples at the boundaries of the segments.", d)
        for seg in segs:
            outs.append(
                {"start": max(last_end, seg["start"] - d), "end": seg["end"] + d}
            )
            last_end = outs[-1]["end"]
        return outs

    if params.keep_silence_in_segments:
        transcribe.get_vad_segments = new_get_vad_segments

    # Move audio to CPU and convert to numpy for the whisper library
    vocals = vocals.cpu().numpy()[0]

    this_duration = vocals.shape[-1] / sr
    logger.debug("Transcribing block in %s, of duration %.1f", language, this_duration)
    
    # Use VAD for longer files to improve accuracy and speed
    use_vad = "auditok" if this_duration > 10 else None

    pipe_output = whisper.transcribe(
        w_model,
        vocals,
        language=language,
        vad=use_vad,
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    # Restore original VAD function
    transcribe.get_vad_segments = old_get_vad_segments

    chunks = []
    for segment in pipe_output.get("segments", []):
        if "words" not in segment:
            logger.warning("No words in segment for %s: %r", in_file, segment)
            continue
        for word in segment["words"]:
            try:
                chunks.append(
                    {"text": word["text"], "timestamp": (word["start"], word["end"])}
                )
            except KeyError:
                logger.error("Missing key in word data for %s: %r", in_file, word)
                raise

    outputs = {
        "alignments": [
            [chunk["text"], chunk["timestamp"], "SPEAKER_MAIN"] for chunk in chunks
        ]
    }
    logger.debug("Whisper transcription applied.")
    with write_and_rename(out_file, "w", pid=True) as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    logger.debug("Wrote file %s", out_file)


def run(params: "Params"):
    """
    Main execution function that sets up models and iterates through audio files.
    """
    init_logging(params.verbose)
    
    # Setup GPU device
    local_rank = 0
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        return
    torch.cuda.set_device(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["OMP_NUM_THREADS"] = "2"

    logger.info("Loading all models.")
    device = torch.device(f"cuda:{local_rank}")
    try:
        w_model = whisper.load_model(params.whisper_model, device=device)
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{params.whisper_model}': {e}")
        return

    logger.info("Loading egs %s.", params.egs)
    try:
        paths = load_audio_paths(params.egs)
    except FileNotFoundError:
        logger.error(f"EGS file not found at {params.egs}. Please check the path.")
        return
        
    logger.info("Processing %d files.", len(paths))

    for idx, path in enumerate(paths):
        print(f"--> Processing file {idx + 1}/{len(paths)}: {path.name}", end='\r')
        out_file = path.with_suffix(".json")
        err_file = path.with_suffix(".json.err")

        if out_file.exists():
            logger.debug("Output file %s already exists, skipping.", out_file)
            continue
        if err_file.exists() and not params.rerun_errors:
            logger.debug("Error file %s exists, skipping.", err_file)
            continue

        try:
            if not path.exists() or path.stat().st_size < 100:
                logger.warning("File is missing or too small: %s", path)
                err_file.touch()
                continue

            logger.debug("Processing file %s -> %s", path, out_file)
            process_one(
                path,
                out_file,
                language=params.lang,
                w_model=w_model,
                params=params,
            )
        except Exception as err:
            # Raise CUDA errors as they might indicate a critical problem
            if "cuda" in repr(err).lower():
                raise
            logger.exception("Error processing %s", path)
            err_file.touch()
            continue
    print(f"\nProcessing finished for {len(paths)} files.")