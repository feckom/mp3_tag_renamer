P3 Identifier + Tagger + Snake-Case Renamer + Lyrics Fetcher (Windows-friendly)

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
