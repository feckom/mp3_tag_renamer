#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP3 Identifier + Tagger + Snake-Case Renamer + Lyrics Fetcher (Windows-friendly)

WHAT IT DOES
- Scans .mp3 files in the current directory (optionally recursively).
- Identifies Artist & Title via Shazam (shazamio, no API key).
- Writes proper ID3 tags (Artist/Title) with correct capitalization (from Shazam).
- RENAMES files to lowercase snake_case with a SHARED STEM for MP3 and LRC:
    e.g.,  marilyn_manson_rock_is_dead.mp3  and  marilyn_manson_rock_is_dead.lrc
- Fetches lyrics from LRCLIB (no key). Writes .lrc (synced if available, else wrapped unsynced).
- DRY-RUN by default; apply real changes with --apply.
- Verbose debug by default; use --quiet to reduce output.

NOTES
- Python 3.11+
- Tested with shazamio 0.8.x, mutagen 1.47+, aiohttp 3.12+
- Having ffmpeg on PATH can improve recognition robustness on Windows.

"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import aiohttp
from mutagen.id3 import ID3, TIT2, TPE1, ID3NoHeaderError
from mutagen.mp3 import MP3
from shazamio import Shazam


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Identify MP3s (Shazam), tag ID3 (proper case), rename to lowercase snake_case, and fetch LRC lyrics.",
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
    p.add_argument("--glob", default="*.mp3", help="Glob to match MP3 files.")
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
    return p.parse_args()


# ------------------------ Logging/Utils ------------------------

def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


BAD_ARTISTS = {
    # Common junky tokens accidentally parsed as artist from scraped filenames
    "been", "brian", "chris", "cloudy", "consumer", "cyberpunk", "gmv", "hanako", "johnny",
    "jvzel", "kang", "outsider", "p", "rat", "rosa", "samurai", "scavenger", "the", "v",
    "you", "yugen"
}


def looks_bad_tag(artist: Optional[str], title: Optional[str]) -> bool:
    """
    Heuristic to decide whether existing ID3 tags are likely junk and should be replaced.
    """
    if not artist or not title:
        return True

    a = artist.strip()
    t = title.strip()

    # suspicious artist: one lowercase token or in known junk set
    if re.fullmatch(r"[a-z]{2,}", a) and a in BAD_ARTISTS:
        return True
    if len(a) <= 3 and a.islower():
        return True

    # suspicious title: contains youtube-like id at end or typical scrap
    if re.search(r"\[[A-Za-z0-9_-]{8,12}\]$", t):
        return True
    if "cyberpunk 2077 ost" in t.lower():
        return True
    if re.fullmatch(r"[a-z0-9_]{20,}", t):
        return True

    return False


def strip_noise_for_titles(s: str) -> str:
    """
    Clean marketing/ID noise while preserving semantic title words.
    Used before turning into the file stem (NOT for ID3 pretty tags).
    """
    if not s:
        return s
    # normalize separators visually
    s = s.replace("｜", " ").replace("／", "/").replace("–", "-").replace("—", "-")

    # drop [YouTubeID], (Official ...), (Music Video), (OST), etc.
    s = re.sub(r"\s*(\(|\[).*?(Official.*?|Music\s*Video|Video|Audio|OST|HD|MV|Lyric.*?|[\w-]{8,12})\s*(\)|\])",
               "", s, flags=re.I)

    # drop remix/edit/nightcore qualifiers for filename readability
    s = re.sub(r"\s*[\(\[][^)\]]*(Remix|Edit|Mix|Nightcore)[^)\]]*[\)\]]", "", s, flags=re.I)

    return s.strip()


def to_snake_component(s: str) -> str:
    """
    Convert a string to lowercase snake_case suitable for a filename component:
    - remove trailing feat. ... for stability
    - collapse non-alphanumeric to underscores
    - collapse multiple underscores, trim edges
    - keep Unicode letters (Windows supports UTF-16 filenames)
    """
    if not s:
        return s

    # remove trailing feat/ft
    s = re.sub(r"\s*[\(\[]?(feat\.?|ft\.?)\s+.+?[\)\]]?$", "", s, flags=re.I)

    # replace separators/punctuation with underscores
    s = strip_noise_for_titles(s)
    # Allow letters/digits; everything else -> underscore
    s = re.sub(r"[^0-9A-Za-z\u00C0-\uFFFF]+", "_", s, flags=re.UNICODE)

    # lowercase, collapse underscores, trim
    s = s.lower()
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def ensure_unique_stem(directory: Path, base_stem: str, exts: Tuple[str, ...]) -> str:
    """
    Ensure the stem is unique across the given extensions within directory.
    If conflicts exist, append _2, _3, ... until free for all extensions.
    """
    stem = base_stem
    n = 2
    while True:
        conflict = False
        for ext in exts:
            if (directory / f"{stem}.{ext}").exists():
                conflict = True
                break
        if not conflict:
            return stem
        stem = f"{base_stem}_{n}"
        n += 1


def planned_unique_stem_for_both(directory: Path, desired_stem: str, lyrics_ext: str) -> str:
    """
    Determine a unique stem that is free for both MP3 and lyrics files.
    """
    return ensure_unique_stem(directory, desired_stem, (lyrics_ext.lower(), "mp3"))


# ------------------------ ID3 Helpers ------------------------

def get_id3(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        audio = ID3(path)
    except ID3NoHeaderError:
        return None, None
    artist = str(audio.get("TPE1").text[0]) if audio.get("TPE1") else None
    title = str(audio.get("TIT2").text[0]) if audio.get("TIT2") else None
    return artist, title


def set_id3(path: Path, artist: str, title: str, quiet: bool, apply: bool) -> None:
    """
    Write proper-cased ID3 tags exactly as provided (from Shazam or trusted source).
    """
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
        log(f"[TAG ] {'WRITE' if apply else 'WOULD WRITE'}  {path.name}  -> "
            f"Artist='{artist_tag}', Title='{title_tag}'", quiet)
    except Exception as e:
        log(f"[TAG ] ERROR writing tags for {path.name}: {e}", quiet)


# ------------------------ LRCLIB (lyrics) ------------------------

def _slim_title(t: str) -> str:
    # remove trailing feat/ft and remix-like qualifiers
    t = re.sub(r"\s*[\(\[]?(feat\.?|ft\.?)\s+.+?[\)\]]?$", "", t, flags=re.I)
    t = re.sub(r"\s*[\(\[][^)\]]*(Remix|Edit|Mix|Nightcore)[^)\]]*[\)\]]$", "", t, flags=re.I)
    return t.strip()


async def _lrclib_query(session: aiohttp.ClientSession, artist: str, title: str) -> Optional[str]:
    params = {"track_name": title, "artist_name": artist, "limit": "1"}
    try:
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


async def lrclib_fetch(session: aiohttp.ClientSession, artist: str, title: str) -> Optional[str]:
    # Try exact first
    lyrics = await _lrclib_query(session, artist, title)
    if not lyrics:
        # Try slimmed title
        lyrics = await _lrclib_query(session, artist, _slim_title(title))
    if not lyrics:
        return None

    # If looks like LRC already, return as-is
    if "[00:" in lyrics or re.search(r"^\[\d{2}:\d{2}", lyrics, flags=re.M):
        return lyrics.strip()

    # Otherwise wrap plain text so it's at least valid .lrc (unsynced)
    lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    return "\n".join(f"[00:00.00]{l}" for l in lines)


def write_lyrics(out_path: Path, lrc_text: str, quiet: bool, apply: bool) -> None:
    if apply:
        out_path.write_text(lrc_text, encoding="utf-8", newline="\n")
    lines = len(lrc_text.splitlines())
    log(f"[LYR ] {'WRITE' if apply else 'WOULD WRITE'}  {out_path.name}  ({lines} lines)", quiet)


# ------------------------ Shazam Recognition ------------------------

async def recognize_with_shazam(shazam: Shazam, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (artist, title) from Shazam or (None, None) if not recognized.
    Proper capitalization is preserved for ID3 tags.
    """
    try:
        res = await shazam.recognize(str(file_path))  # modern API
        track = res.get("track") if isinstance(res, dict) else None
        if not track:
            return None, None
        title = (track.get("title") or "").strip()
        artist = (track.get("subtitle") or "").strip()
        if artist and title:
            return artist, title
    except Exception:
        pass
    return None, None


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
) -> None:
    log(f"\n[FILE] {path.name}", quiet)

    # Validate MP3
    try:
        _ = MP3(path)
    except Exception:
        log("[WARN] Not a valid MP3 or unsupported. Skipping.", quiet)
        return

    # Size hint
    try:
        size_b = path.stat().st_size
        if size_b < min_bytes:
            log(f"[WARN] File is quite small ({size_b} B). Recognition may be unreliable.", quiet)
    except Exception:
        pass

    # Existing tags
    cur_artist, cur_title = get_id3(path)
    if cur_artist and cur_title:
        log(f"[TAG ] Existing tags: Artist='{cur_artist}' Title='{cur_title}'", quiet)

    # Decide recognition
    if tags_policy == "overwrite":
        need_recognize = True
    elif tags_policy == "keep":
        need_recognize = not (cur_artist and cur_title) or force
    else:  # 'prefer'
        need_recognize = force or looks_bad_tag(cur_artist, cur_title)

    artist, title = (cur_artist, cur_title)

    if need_recognize:
        log("[NET ] Shazam recognize...", quiet)
        artist, title = await recognize_with_shazam(shazam, path)
        await asyncio.sleep(max(0, rate_ms) / 1000.0)
        if artist and title:
            log(f"[NET ] Shazam: Artist='{artist}' Title='{title}'", quiet)
        else:
            log("[NET ] Shazam failed to identify.", quiet)

    if not (artist and title):
        log("[STOP] No tags to write (artist/title unresolved).", quiet)
        return

    # Write/refresh proper-cased tags if changed or forced
    if force or (cur_artist != artist or cur_title != title):
        set_id3(path, artist, title, quiet, apply)
    else:
        log("[TAG ] Up-to-date; no write needed.", quiet)

    # Build desired lowercase snake_case STEM (same for .mp3 and .lrc)
    # We turn pretty tags → cleaned → snake components
    snake_artist = to_snake_component(artist)
    snake_title = to_snake_component(title)
    desired_stem = f"{snake_artist}_{snake_title}".strip("_")

    # Determine a unique stem free for BOTH {stem}.mp3 and {stem}.{ext}
    directory = path.parent
    final_stem = planned_unique_stem_for_both(directory, desired_stem, lyrics_ext)

    # Plan final paths
    planned_mp3 = directory / f"{final_stem}.mp3"
    planned_lrc = directory / f"{final_stem}.{lyrics_ext}"

    # RENAME MP3 to final stem (or show what would happen)
    if path.resolve() != planned_mp3.resolve():
        if apply:
            # If target exists, ensure we use our chosen unique stem
            # (planned_* already reflects a unique stem).
            # Perform rename:
            if planned_mp3.exists():
                # Should not happen thanks to unique stem, but guard anyway
                log(f"[MOVE] COLLISION at planned target (unexpected): {planned_mp3.name}", quiet)
                # Fallback: append a microtime hash
                micro = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
                final_stem = f"{final_stem}_{micro}"
                planned_mp3 = directory / f"{final_stem}.mp3"
                planned_lrc = directory / f"{final_stem}.{lyrics_ext}"

            path.rename(planned_mp3)
            log(f"[MOVE] RENAME  {path.name}  ->  {planned_mp3.name}", quiet)
            final_mp3_path = planned_mp3
        else:
            log(f"[MOVE] WOULD RENAME  {path.name}  ->  {planned_mp3.name}", quiet)
            final_mp3_path = planned_mp3  # imagined future path for consistent LRC naming
    else:
        log(f"[MOVE] SKIP (already has desired name) {path.name}", quiet)
        final_mp3_path = path

    # LYRICS (save under the SAME STEM as MP3)
    if not no_lyrics:
        if planned_lrc.exists() and not force:
            log(f"[LYR ] Exists, skip: {planned_lrc.name}", quiet)
        else:
            log("[NET ] LRCLIB query...", quiet)
            lrc = await lrclib_fetch(session, artist, title)
            await asyncio.sleep(max(0, rate_ms) / 1000.0)
            if lrc:
                # Write exactly to planned_lrc (same stem as final MP3)
                write_lyrics(planned_lrc, lrc, quiet, apply)
            else:
                log("[LYR ] No lyrics found.", quiet)


async def main_async(args: argparse.Namespace) -> int:
    # Find files
    base = Path.cwd()
    if args.recursive:
        files: List[Path] = [p for p in base.rglob(args.glob) if p.is_file()]
    else:
        files = [p for p in base.glob(args.glob) if p.is_file()]

    if not files:
        log("[INFO] No MP3 files matched. Check working directory or --glob.", args.quiet)
        return 0

    log(f"[INFO] Found {len(files)} file(s). Dry-run={not args.apply}, Debug={not args.quiet}", args.quiet)

    timeout = aiohttp.ClientTimeout(total=90)
    connector = aiohttp.TCPConnector(limit=4)
    shazam = Shazam()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for path in files:
            try:
                await process_file(
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
                )
            except KeyboardInterrupt:
                print("\n[INT ] Interrupted by user.")
                return 130
            except Exception as e:
                log(f"[ERR ] Unexpected error for {path.name}: {e}", args.quiet)

    log("\n[DONE] Finished.", args.quiet)
    return 0


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

