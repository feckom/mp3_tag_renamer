#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP3 Identifier + Tagger + Snake-Case Renamer + Lyrics Fetcher + LRC Fixer via Whisper (Windows-friendly)
Overwrite mode: no suffixes; always overwrite target MP3/LRC if they already exist.

DEBUG ENHANCEMENTS
- Startup summary of args and environment (ffmpeg path, backend, device, threads)
- Per-file detailed decisions with reasons (why recognize, why rename, why run Whisper)
- LRC diagnostics: timestamp count, unique/zero ratios, first N samples
- Whisper diagnostics: backend, model, timings, segment counts, first N segments preview
- Write/overwrite decisions explicitly logged (with reason)
- Errors logged with exception type and message

NOTES
- Python 3.11+
- Install: shazamio, mutagen, aiohttp, faster-whisper (or openai-whisper), ffmpeg in PATH
- Optional: tqdm (for progress bar), psutil (for disk space check)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set, Callable, Awaitable
import shutil  # still needed for ffmpeg which check

import aiohttp
from mutagen.id3 import ID3, TIT2, TPE1, ID3NoHeaderError
from mutagen.mp3 import MP3
from shazamio import Shazam

# Optional imports
try:
    from tqdm.asyncio import tqdm
except ImportError:
    tqdm = None

try:
    import psutil
except ImportError:
    psutil = None

# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Identify audio files (Shazam), tag ID3, rename to lowercase snake_case, fetch/repair LRC lyrics (overwrite).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run).")
    p.add_argument("--recursive", "-r", action="store_true", help="Scan subfolders recursively.")
    p.add_argument("--force", action="store_true", help="Force overwrite of existing tags/lyrics.")
    p.add_argument("--quiet", "-q", action="store_true", help="Reduce output (verbose by default).")
    p.add_argument("--min-bytes", type=int, default=160_000,
                   help="Minimum file size to attempt recognition (skip tiny files).")
    p.add_argument("--rate-ms", type=int, default=1200,
                   help="Sleep between network recognitions/queries (ms) to avoid throttling).")
    p.add_argument("--no-lyrics", action="store_true", help="Do not fetch lyrics.")
    p.add_argument("--ext", default="lrc", help="Lyrics file extension to write.")
    p.add_argument("--audio-formats", default="mp3", help="Comma-separated audio extensions (e.g., mp3,flac,m4a).")
    p.add_argument(
        "--tags-policy",
        choices=("keep", "prefer", "overwrite"),
        default="prefer",
        help=(
            "Tag policy: 'keep' = never recognize if both tags exist; "
            "'prefer' = recognize when existing tags look suspicious; "
            "'overwrite' = always recognize and overwrite."
        ),
    )
    # Whisper controls
    p.add_argument("--whisper-mode", choices=("off", "if-bad", "always"), default="if-bad",
                   help="Use Whisper to generate timed LRC: off / only if missing or bad / always.")
    p.add_argument("--whisper-backend", choices=("whisper", "faster"), default="faster",
                   help="Backend: 'whisper' (openai-whisper) or 'faster' (faster-whisper).")
    p.add_argument("--whisper-model", default="medium",
                   help="Model name (e.g., tiny, base, small, medium, large-v3).")
    p.add_argument("--whisper-language", default="auto", choices=("auto", "sk", "cs", "en"),
                   help="Language hint for transcription.")
    p.add_argument("--whisper-device", default=None,
                   help="Device for inference (e.g., 'cuda', 'cpu').")
    p.add_argument("--whisper-cpu-threads", type=int, default=0,
                   help="CPU threads for Whisper (0 = library default).")
    p.add_argument("--whisper-translate", action="store_true",
                   help="Translate to English while keeping timing (usually OFF for lyrics).")
    p.add_argument("--debug-preview", type=int, default=4,
                   help="How many timestamps/segments to preview in debug logs.")
    # Performance
    p.add_argument("--max-concurrent", type=int, default=4,
                   help="Max concurrent file processing (network + whisper).")
    return p.parse_args()

# ------------------------ Logging/Utils ------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)

def dbg(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[DBG { _ts() }] {msg}")

def warn(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[WARN] {msg}")

def err(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[ERR ] {msg}")

class Timer:
    def __init__(self, label: str, quiet: bool):
        self.label = label
        self.quiet = quiet
        self.t0 = time.perf_counter()
    def stop(self, extra: str = ""):
        dt = (time.perf_counter() - self.t0) * 1000.0
        dbg(f"{self.label} took {dt:.1f} ms{(' - ' + extra) if extra else ''}", self.quiet)

BAD_ARTISTS = {
    "been", "brian", "chris", "cloudy", "consumer", "cyberpunk", "gmv", "hanako", "johnny",
    "jvzel", "kang", "outsider", "p", "rat", "rosa", "samurai", "scavenger", "the", "v",
    "you", "yugen"
}

def looks_bad_tag(artist: Optional[str], title: Optional[str]) -> bool:
    if not artist or not title:
        return True
    a = artist.strip()
    t = title.strip()
    if re.fullmatch(r"[a-z]{2,}", a) and a in BAD_ARTISTS:
        return True
    if len(a) <= 3 and a.islower():
        return True
    if re.search(r"\[[A-Za-z0-9_-]{8,12}\]$", t):
        return True
    if "cyberpunk 2077 ost" in t.lower():
        return True
    if re.fullmatch(r"[a-z0-9_]{20,}", t):
        return True
    return False

def clean_title_for_search(s: str) -> str:
    """ Unified function for cleaning titles for both renaming and LRCLIB search. """
    if not s:
        return s
    s = s.replace("｜", " ").replace("／", "/").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*(\(|\[).*?(Official.*?|Music\s*Video|Video|Audio|OST|HD|MV|Lyric.*?|[\w-]{8,12})\s*(\)|\])",
               "", s, flags=re.I)
    s = re.sub(r"\s*[\(\[][^)\]]*(Remix|Edit|Mix|Nightcore)[^)\]]*[\)\]]", "", s, flags=re.I)
    s = re.sub(r"\s*[\(\[]?(feat\.?|ft\.?)\s+.+?[\)\]]?$", "", s, flags=re.I)
    return s.strip()

def to_snake_component(s: str) -> str:
    if not s:
        return s
    s = clean_title_for_search(s)
    s = re.sub(r"[^0-9A-Za-z\u00C0-\uFFFF]+", "_", s, flags=re.UNICODE)
    s = s.lower()
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# ------------------------ ID3 Helpers ------------------------

def get_id3(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        audio = ID3(path)
    except ID3NoHeaderError:
        return None, None
    artist_frame = audio.get("TPE1")
    title_frame = audio.get("TIT2")
    artist = str(artist_frame.text[0]) if artist_frame and len(artist_frame.text) > 0 else None
    title = str(title_frame.text[0]) if title_frame and len(title_frame.text) > 0 else None
    return artist, title

def set_id3(path: Path, artist: str, title: str, quiet: bool, apply: bool) -> None:
    artist_tag = (artist or "").strip()
    title_tag = (title or "").strip()
    try:
        try:
            tags = ID3(path)
        except ID3NoHeaderError:
            tags = ID3()
        tags["TPE1"] = TPE1(encoding=3, text=artist_tag)
        tags["TIT2"] = TIT2(encoding=3, text=title_tag)
        if apply:
            tags.save(path)
        log(f"[TAG ] {'WRITE' if apply else 'WOULD WRITE'}  {path.name}  -> Artist='{artist_tag}', Title='{title_tag}'", quiet)
    except Exception as e:
        err(f"[TAG ] ERROR writing tags for {path.name}: {type(e).__name__}: {e}", quiet)
        dbg(traceback.format_exc(), quiet)

# ------------------------ LRCLIB (lyrics) ------------------------

async def _lrclib_query(session: aiohttp.ClientSession, artist: str, title: str) -> Optional[str]:
    params = {"track_name": title, "artist_name": artist, "limit": "1"}
    try:
        # FIXED: removed trailing spaces in URL
        async with session.get("https://lrclib.net/api/search", params=params, timeout=20) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None
    if not data:
        return None
    item = data[0]
    return item.get("syncedLyrics") or item.get("plainLyrics") or item.get("unsyncedLyrics")

async def lrclib_fetch(session: aiohttp.ClientSession, artist: str, title: str, quiet: bool) -> Optional[str]:
    tmr = Timer("LRCLIB query", quiet)
    lyrics = await _lrclib_query(session, artist, title)
    if not lyrics:
        lyrics = await _lrclib_query(session, artist, clean_title_for_search(title))
    tmr.stop(f"artist='{artist}', title='{title}'")
    if not lyrics:
        return None
    # already in LRC?
    if "[00:" in lyrics or re.search(r"^\[\d{2}:\d{2}", lyrics, flags=re.M):
        dbg("LRCLIB returned LRC format", quiet)
        return lyrics.strip()
    lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    dbg(f"LRCLIB returned UNSYNCED lyrics; wrapping {len(lines)} lines at [00:00.00]", quiet)
    return "\n".join(f"[00:00.00]{l}" for l in lines)

def write_lyrics(out_path: Path, lrc_text: str, quiet: bool, apply: bool) -> None:
    if apply:
        try:
            out_path.write_text(lrc_text, encoding="utf-8", newline="\n")  # overwrite
            written_size = len(lrc_text.encode("utf-8"))
            dbg(f"Wrote {written_size} bytes to {out_path}", quiet)
        except Exception as e:
            raise RuntimeError(f"Failed to write lyrics to {out_path}: {e}")
    lines = len(lrc_text.splitlines())
    log(f"[LYR ] {'WRITE' if apply else 'WOULD WRITE'}  {out_path.name}  ({lines} lines)", quiet)

# ------------------------ Shazam Recognition ------------------------

async def recognize_with_shazam(shazam: Shazam, file_path: Path, quiet: bool) -> Tuple[Optional[str], Optional[str]]:
    dbg(f"Shazam recognizing: {file_path.name}", quiet)
    tmr = Timer("Shazam recognize", quiet)
    try:
        res = await shazam.recognize(str(file_path))
        track = res.get("track") if isinstance(res, dict) else None
        if not track:
            tmr.stop("no track")
            return None, None
        title = (track.get("title") or "").strip()
        artist = (track.get("subtitle") or "").strip()
        tmr.stop(f"got: artist='{artist}' title='{title}'")
        if artist and title:
            return artist, title
    except Exception as e:
        tmr.stop("exception")
        err(f"[NET ] Shazam error: {type(e).__name__}: {e}", quiet)
        dbg(traceback.format_exc(), quiet)
    return None, None

# ------------------------ LRC Quality Heuristics ------------------------

# [MM:SS.xx] or [MM:SS,xx] or [MM:SS]
LRC_TIME_RE = re.compile(r"^\[(\d{2}):(\d{2})(?:[.,](\d{2,3}))?\]", re.M)

def parse_lrc_times(lrc_text: str) -> List[Tuple[int,int,int]]:
    """
    Returns list of (mm, ss, cs) tuples for each timestamp found.
    cs = centiseconds (00..99) or 0 if missing.
    """
    out: List[Tuple[int,int,int]] = []
    if not lrc_text:
        return out
    for m in LRC_TIME_RE.finditer(lrc_text):
        mm = int(m.group(1))
        ss = int(m.group(2))
        cs_raw = m.group(3)
        if cs_raw is None:
            cs = 0
        else:
            cs = int(cs_raw[:2].ljust(2, "0"))
        out.append((mm, ss, cs))
    return out

def _lrc_stats(lrc_text: str, preview: int) -> str:
    ts = parse_lrc_times(lrc_text)
    n = len(ts)
    uniq = len(set(ts))
    zeros = sum(1 for (m, s, _) in ts if m == 0 and s == 0)
    prev = ", ".join([f"{mm:02d}:{ss:02d}.{cs:02d}" for mm, ss, cs in ts[:preview]])
    return f"timestamps={n}, unique={uniq}, zeros={zeros}, sample=[{prev}]"

def is_broken_lrc(lrc_text: Optional[str]) -> bool:
    """
    Broken if:
    - No timestamps at all, or
    - All timestamps are the same, or
    - >=60% timestamps are 00:00.xx
    """
    if not lrc_text:
        return True
    ts = parse_lrc_times(lrc_text)
    if not ts:
        return True
    if len({t for t in ts}) <= 1:
        return True
    zeros = sum(1 for (m,s,_) in ts if m == 0 and s == 0)
    return zeros >= max(3, int(0.6 * len(ts)))

# ------------------------ Whisper Transcription → LRC ------------------------

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def ffmpeg_path() -> Optional[str]:
    p = shutil.which("ffmpeg")
    return p

def format_ts(seconds: float) -> str:
    if seconds < 0: seconds = 0.0
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100.0))
    if cs >= 100:
        ss += 1
        cs -= 100
    return f"[{mm:02d}:{ss:02d}.{cs:02d}]"

def build_lrc_from_segments(segments: List[Tuple[float, float, str]], preview: int, quiet: bool) -> str:
    lines: List[str] = []
    for i, (start, _, text) in enumerate(segments):
        text = (text or "").strip()
        if not text:
            continue
        if i < preview:
            dbg(f"SEG[{i}] {format_ts(start)} {text[:80]!r}", quiet)
        lines.append(f"{format_ts(start)}{text}")
    return "\n".join(lines)

# Cache for Whisper model to load once
class WhisperModelCache:
    _instance: Optional[Any] = None
    _backend: Optional[str] = None
    _model_name: Optional[str] = None
    _device: Optional[str] = None
    _compute_type: Optional[str] = None
    _cpu_threads: int = 0

    @classmethod
    def get_model(cls, *, backend: str, model: str, device: Optional[str], cpu_threads: int) -> Any:
        compute_type = "float16" if device == "cuda" else "int8"
        should_reload = (
            cls._instance is None or
            cls._backend != backend or
            cls._model_name != model or
            cls._device != device or
            cls._compute_type != compute_type or
            cls._cpu_threads != cpu_threads
        )
        if should_reload:
            cls._unload()
            cls._load(backend, model, device, compute_type, cpu_threads)
        return cls._instance

    @classmethod
    def _load(cls, backend: str, model: str, device: Optional[str], compute_type: str, cpu_threads: int):
        dbg(f"Loading Whisper model: {model} ({backend}) on {device or 'cpu'}", False)
        if backend == "whisper":
            import whisper
            opts = {}
            if device:
                opts["device"] = device
            if cpu_threads > 0:
                os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
                os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
            cls._instance = whisper.load_model(model, **opts)
        else:  # faster-whisper
            from faster_whisper import WhisperModel
            cls._instance = WhisperModel(
                model,
                device=(device or "cpu"),
                compute_type=compute_type,
                cpu_threads=cpu_threads or 0,
                # num_workers=1  # for future batch processing
            )
        cls._backend = backend
        cls._model_name = model
        cls._device = device
        cls._compute_type = compute_type
        cls._cpu_threads = cpu_threads

    @classmethod
    def _unload(cls):
        if cls._instance:
            del cls._instance
            cls._instance = None

def transcribe_with_whisper(
    audio_path: Path,
    *,
    backend: str,
    model: str,
    language: str,
    device: Optional[str],
    cpu_threads: int,
    translate: bool,
    preview: int,
    quiet: bool,
) -> Optional[str]:
    """
    Returns LRC text built from Whisper segments, or None if unavailable.
    Uses cached model if possible.
    """
    dbg(f"Whisper plan: backend={backend}, model={model}, lang={language}, device={device}, threads={cpu_threads}, translate={translate}", quiet)

    if not have_ffmpeg():
        err("[WSPR] ffmpeg not found in PATH; cannot run Whisper timing.", quiet)
        return None
    else:
        dbg(f"ffmpeg at: {ffmpeg_path()}", quiet)

    # Validate model (basic known models)
    KNOWN_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    if model not in KNOWN_MODELS:
        warn(f"Unknown Whisper model: '{model}'. May fail to load.", quiet)

    try:
        wmodel = WhisperModelCache.get_model(
            backend=backend,
            model=model,
            device=device,
            cpu_threads=cpu_threads
        )

        lang = None if language == "auto" else language
        task = "translate" if translate else "transcribe"

        if backend == "whisper":
            t_trans = Timer("Whisper transcribe", quiet)
            result = wmodel.transcribe(
                str(audio_path),
                language=lang,
                task=task,
                verbose=False,
                condition_on_previous_text=False,
                fp16=(device == "cuda"),
            )
            t_trans.stop()
            segs = []
            for seg in result.get("segments", []):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                text = str(seg.get("text", "")).strip()
                segs.append((start, end, text))
            dbg(f"Whisper segments: {len(segs)}", quiet)
            return build_lrc_from_segments(segs, preview, quiet) if segs else None

        else:  # faster-whisper
            t_trans = Timer("faster-whisper transcribe", quiet)
            segments, _info = wmodel.transcribe(
                str(audio_path),
                language=lang,
                task=task,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200},
                condition_on_previous_text=False,
                beam_size=1,
                # batch_size=16  # for future batch processing
            )
            t_trans.stop()
            segs = []
            count = 0
            for seg in segments:
                start = float(seg.start or 0.0)
                end = float(seg.end or start)
                text = (seg.text or "").strip()
                segs.append((start, end, text))
                count += 1
            dbg(f"Whisper segments: {count}", quiet)
            return build_lrc_from_segments(segs, preview, quiet) if segs else None

    except Exception as e:
        err(f"[WSPR] Whisper error: {type(e).__name__}: {e}", quiet)
        dbg(traceback.format_exc(), quiet)
        return None

# ------------------------ Disk Space Check ------------------------

def check_disk_space(path: Path, required_bytes: int, quiet: bool) -> bool:
    if psutil is None:
        return True  # skip check if psutil not available
    try:
        partition = psutil.disk_usage(path.parent.resolve())
        free_mb = partition.free / (1024**2)
        required_mb = required_bytes / (1024**2)
        if partition.free < required_bytes:
            err(f"Insufficient disk space: need {required_mb:.1f}MB, have {free_mb:.1f}MB on {path.parent}", quiet)
            return False
        dbg(f"Disk space OK: {free_mb:.1f}MB free, need {required_mb:.1f}MB", quiet)
        return True
    except Exception as e:
        warn(f"Could not check disk space: {e}", quiet)
        return True

# ------------------------ Core Pipeline ------------------------

async def process_file(
    path: Path,
    *,
    apply: bool,
    quiet: bool,
    force: bool,
    no_lyrics: bool,
    lyrics_ext: str,
    shazam: Shazam,
    session: aiohttp.ClientSession,
    rate_ms: int,
    tags_policy: str,
    min_bytes: int,
    whisper_mode: str,
    whisper_backend: str,
    whisper_model: str,
    whisper_language: str,
    whisper_device: Optional[str],
    whisper_cpu_threads: int,
    whisper_translate: bool,
    debug_preview: int,
) -> Tuple[Path, Optional[Exception]]:
    """
    Returns (original_path, error) if failed, or (final_mp3_path, None) if ok.
    """
    try:
        log(f"\n[FILE] {path.name}", quiet)

        # Validate audio (MP3 or others via mutagen)
        try:
            audio = MP3(path)  # for now, only MP3 supported for tag writing
            dbg(f"Audio ok: length={getattr(audio.info, 'length', 0):.2f}s bitrate={getattr(audio.info, 'bitrate', 0)}", quiet)
        except Exception as e:
            warn(f"Not a valid MP3 or unsupported ({type(e).__name__}: {e}). Skipping.", quiet)
            return path, None

        # Size hint
        try:
            size_b = path.stat().st_size
            dbg(f"File size: {size_b} B", quiet)
            if size_b < min_bytes:
                warn(f"File is small ({size_b} B). Recognition may be unreliable.", quiet)
        except Exception as e:
            warn(f"stat() failed: {e}", quiet)

        # Existing tags
        cur_artist, cur_title = get_id3(path)
        dbg(f"Existing tags: artist={cur_artist!r}, title={cur_title!r}", quiet)

        # Decide recognition
        if tags_policy == "overwrite":
            need_recognize = True
            reason = "tags-policy=overwrite"
        elif tags_policy == "keep":
            need_recognize = not (cur_artist and cur_title) or force
            reason = "no tags or --force" if need_recognize else "keep existing"
        else:  # 'prefer'
            need_recognize = force or looks_bad_tag(cur_artist, cur_title)
            reason = "--force or looks_bad_tag" if need_recognize else "existing look OK"

        dbg(f"Recognize decision: need_recognize={need_recognize} ({reason})", quiet)

        artist, title = (cur_artist, cur_title)

        if need_recognize:
            log("[NET ] Shazam recognize...", quiet)
            artist, title = await recognize_with_shazam(shazam, path, quiet)
            await asyncio.sleep(max(0, rate_ms) / 1000.0)
            if artist and title:
                log(f"[NET ] Shazam: Artist='{artist}' Title='{title}'", quiet)
            else:
                warn("Shazam failed to identify; will keep existing tags if any.", quiet)

        if not (artist and title):
            log("[STOP] No tags to write (artist/title unresolved).", quiet)
            return path, None

        # Write/refresh proper-cased tags if changed or forced
        if force or (cur_artist != artist or cur_title != title):
            set_id3(path, artist, title, quiet, apply)
        else:
            log("[TAG ] Up-to-date; no write needed.", quiet)

        # Build lowercase snake_case STEM (same for .mp3 and .lrc)
        snake_artist = to_snake_component(artist)
        snake_title = to_snake_component(title)
        final_stem = f"{snake_artist}_{snake_title}".strip("_")
        dbg(f"Planned stem: {final_stem}", quiet)

        directory = path.parent
        planned_mp3 = directory / f"{final_stem}.mp3"
        planned_lrc = directory / f"{final_stem}.{lyrics_ext}"

        # RENAME MP3 — OVERWRITE TARGET if exists
        if path.resolve() != planned_mp3.resolve():
            if apply:
                # Check disk space for rename (copy if cross-device)
                try:
                    if not check_disk_space(planned_mp3, path.stat().st_size, quiet):
                        return path, RuntimeError("Insufficient disk space for rename")
                    tmr = Timer("Rename/overwrite MP3", quiet)
                    path.replace(planned_mp3)  # overwrite atomically
                    tmr.stop()
                    log(f"[MOVE] RENAME  {path.name}  ->  {planned_mp3.name} (overwritten if existed)", quiet)
                    final_mp3_path = planned_mp3
                except Exception as e:
                    raise RuntimeError(f"Failed to rename {path} -> {planned_mp3}: {e}")
            else:
                log(f"[MOVE] WOULD RENAME  {path.name}  ->  {planned_mp3.name} (would overwrite if existed)", quiet)
                final_mp3_path = planned_mp3
        else:
            log(f"[MOVE] SKIP (already has desired name) {path.name}", quiet)
            final_mp3_path = path

        # ---------------- LYRICS FLOW ----------------
        # 1) Fetch existing or from LRCLIB (unless disabled)
        fetched_lrc: Optional[str] = None
        existing_on_disk: Optional[str] = None
        if not no_lyrics:
            if planned_lrc.exists():
                try:
                    existing_on_disk = planned_lrc.read_text(encoding="utf-8", errors="ignore")
                    log(f"[LYR ] Existing LRC on disk: {planned_lrc.name}", quiet)
                    dbg(_lrc_stats(existing_on_disk, debug_preview), quiet)
                except Exception as e:
                    warn(f"Failed to read existing LRC ({type(e).__name__}: {e})", quiet)

            if existing_on_disk is None:
                log("[NET ] LRCLIB query...", quiet)
                fetched_lrc = await lrclib_fetch(session, artist, title, quiet)
                await asyncio.sleep(max(0, rate_ms) / 1000.0)
                if fetched_lrc:
                    log("[LYR ] Got lyrics from LRCLIB.", quiet)
                    dbg(_lrc_stats(fetched_lrc, debug_preview), quiet)
                else:
                    log("[LYR ] No lyrics found from LRCLIB.", quiet)
            else:
                fetched_lrc = existing_on_disk

        # 2) Decide if Whisper should run
        need_whisper = False
        broken_note = ""
        if whisper_mode == "always":
            need_whisper = True
            broken_note = "forced by --whisper-mode=always"
        elif whisper_mode == "if-bad":
            if not fetched_lrc:
                need_whisper = True
                broken_note = "no LRC available"
            else:
                broken = is_broken_lrc(fetched_lrc)
                need_whisper = broken
                broken_note = "broken LRC detected" if broken else "LRC looks OK"
        else:
            broken_note = "disabled (off)"

        log(f"[WSPR] Decision: need_whisper={need_whisper} ({broken_note})", quiet)

        # 3) If Whisper is needed, build timed LRC from audio
        whisper_lrc: Optional[str] = None
        if need_whisper:
            whisper_lrc = transcribe_with_whisper(
                final_mp3_path,
                backend=whisper_backend,
                model=whisper_model,
                language=whisper_language,
                device=whisper_device,
                cpu_threads=whisper_cpu_threads,
                translate=whisper_translate,
                preview=debug_preview,
                quiet=quiet,
            )
            if whisper_lrc:
                log("[WSPR] Built time-aligned LRC from Whisper.", quiet)
                dbg(_lrc_stats(whisper_lrc, debug_preview), quiet)
            else:
                warn("Whisper failed to produce LRC.", quiet)

        # 4) Pick best LRC to write (prefer Whisper if available)
        final_lrc_to_write: Optional[str] = whisper_lrc or fetched_lrc

        # 5) Write/overwrite LRC (or report missing)
        if final_lrc_to_write:
            existing_is_broken = is_broken_lrc(existing_on_disk) if existing_on_disk is not None else True
            should_overwrite = (
                (whisper_lrc is not None) or  # prefer Whisper result
                (existing_on_disk is None) or
                existing_is_broken or
                force
            )
            dbg(f"LRC overwrite decision: should_overwrite={should_overwrite} (whisper={whisper_lrc is not None}, exist={existing_on_disk is not None}, broken={existing_is_broken}, force={force})", quiet)
            if should_overwrite:
                if apply:
                    if not check_disk_space(planned_lrc, len(final_lrc_to_write.encode("utf-8")), quiet):
                        return path, RuntimeError("Insufficient disk space for LRC")
                write_lyrics(planned_lrc, final_lrc_to_write, quiet, apply)
            else:
                log("[LYR ] KEEP existing LRC (looks OK).", quiet)
        else:
            log("[LYR ] No lyrics produced (LRCLIB empty and Whisper failed/off).", quiet)

        return final_mp3_path, None

    except Exception as e:
        return path, e

# ------------------------ Orchestrator ------------------------

async def main_async(args: argparse.Namespace) -> int:
    # Validate args
    if args.min_bytes < 1:
        err("Error: --min-bytes must be >= 1", False)
        return 1
    if args.rate_ms < 0:
        err("Error: --rate-ms must be >= 0", False)
        return 1

    base = Path.cwd()
    extensions = [ext.strip().lower() for ext in args.audio_formats.split(",") if ext.strip()]
    if not extensions:
        err("Error: --audio-formats must contain at least one extension", False)
        return 1

    # Build glob patterns
    globs = [f"*.{ext}" for ext in extensions]
    files: List[Path] = []
    for pattern in globs:
        if args.recursive:
            files.extend([p for p in base.rglob(pattern) if p.is_file()])
        else:
            files.extend([p for p in base.glob(pattern) if p.is_file()])

    if not files:
        log("[INFO] No audio files matched. Check working directory or --audio-formats.", args.quiet)
        return 0

    # Startup environment / settings debug
    dbg(f"ARGS: apply={args.apply}, recursive={args.recursive}, force={args.force}, quiet={args.quiet}", args.quiet)
    dbg(f"WHISPER: mode={args.whisper_mode}, backend={args.whisper_backend}, model={args.whisper_model}, lang={args.whisper_language}, device={args.whisper_device}, threads={args.whisper_cpu_threads}, translate={args.whisper_translate}", args.quiet)
    dbg(f"SYSTEM: ffmpeg={ffmpeg_path() or 'NOT FOUND'}", args.quiet)
    dbg(f"SCAN: files={len(files)}, formats={extensions}, min_bytes={args.min_bytes}", args.quiet)
    log(f"[INFO] Found {len(files)} file(s). Dry-run={not args.apply}, Debug={not args.quiet}", args.quiet)

    # Dry-run: check resources
    if not have_ffmpeg():
        warn("ffmpeg not found in PATH. Whisper will be skipped if needed.", args.quiet)
    else:
        dbg(f"ffmpeg OK at: {ffmpeg_path()}", args.quiet)

    # Check Whisper model availability (if needed)
    if args.whisper_mode != "off":
        KNOWN_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
        if args.whisper_model not in KNOWN_MODELS:
            warn(f"Unknown Whisper model: '{args.whisper_model}'. May fail to load.", args.quiet)

    timeout = aiohttp.ClientTimeout(total=90)
    connector = aiohttp.TCPConnector(limit=4)
    shazam = Shazam()

    # Prepare progress display
    if tqdm and not args.quiet:
        file_iter = tqdm(files, desc="Processing", unit="file")
    else:
        file_iter = files

    # Process files with limited concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)
    errors: List[Tuple[Path, Exception]] = []

    async def process_with_semaphore(path: Path):
        async with semaphore:
            return await process_file(
                path,
                apply=args.apply,
                quiet=args.quiet,
                force=args.force,
                no_lyrics=args.no_lyrics,
                lyrics_ext=args.ext.lower(),
                shazam=shazam,
                session=session,
                rate_ms=args.rate_ms,
                tags_policy=args.tags_policy,
                min_bytes=args.min_bytes,
                whisper_mode=args.whisper_mode,
                whisper_backend=args.whisper_backend,
                whisper_model=args.whisper_model,
                whisper_language=args.whisper_language,
                whisper_device=args.whisper_device,
                whisper_cpu_threads=args.whisper_cpu_threads,
                whisper_translate=args.whisper_translate,
                debug_preview=args.debug_preview,
            )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [process_with_semaphore(path) for path in file_iter]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                err(f"Critical error in task: {result}", args.quiet)
                continue
            if isinstance(result, tuple) and len(result) == 2:
                orig_path, error = result
                if error:
                    err(f"Error processing {orig_path.name}: {type(error).__name__}: {error}", args.quiet)
                    errors.append((orig_path, error))

    if errors:
        log(f"\n[SUMMARY] {len(errors)} file(s) failed:", args.quiet)
        for path, e in errors[:10]:  # show first 10
            err(f"  {path.name}: {e}", args.quiet)
        if len(errors) > 10:
            log(f"  ... and {len(errors)-10} more.", args.quiet)

    log(f"\n[DONE] Finished. {len(files) - len(errors)} succeeded, {len(errors)} failed.", args.quiet)
    return 1 if errors else 0

def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[INT ] Interrupted.")
        code = 130
    sys.exit(code)

if __name__ == "__main__":
    main()
