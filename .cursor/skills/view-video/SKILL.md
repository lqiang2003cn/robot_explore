---
name: view-video
description: >-
  Uploads a local video file to public hosting and returns a browser-playable
  URL. Use when the user says "view video:" followed by a path, or asks to
  share, host, or get a link for a local video file without extra commentary.
---

# View video (host and link)

## Trigger

When the user’s message matches **`view video:`** followed by a file path (with or without spaces after the colon), treat everything after the first `view video:` as the path. Examples:

- `view video:vla_x3plus/output/demo.mp4`
- `view video: /abs/path/to/clip.mp4`

Match case-insensitively on `view video` if the intent is clearly the same.

## Path resolution

1. Strip the `view video:` prefix and trim whitespace from the path token.
2. If the path is not absolute, resolve it relative to the **workspace root**.
3. Verify the file exists and is non-empty (`test -f`, `stat`).

## Upload

Use **Catbox** anonymous upload (MP4 and common video types work; max **200 MB** per file):

```bash
curl -sS -F "reqtype=fileupload" -F "fileToUpload=@/absolute/path/to/file" https://catbox.moe/user/api.php
```

- Use the **absolute** path in `fileToUpload=@...`.
- On success the response body is a single HTTPS URL (e.g. `https://files.catbox.moe/....mp4`).
- If the response starts with `Error` or is not a valid `https://` URL, report the failure briefly (one short sentence) and do not invent a link.

## Response format

- **Success:** Reply with **only** the returned URL on its own line (or a single line). No preamble, no markdown link unless the user already uses rich formatting—default is plain URL.
- **Failure:** One short error line; do not suggest the user run commands themselves unless the environment cannot run `curl`.

## Notes

- Do not upload secrets or private data; hosting is public.
- If `curl` is unavailable or the file exceeds limits, say so briefly and stop.
