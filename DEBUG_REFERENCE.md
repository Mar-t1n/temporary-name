"""
Debug Output Reference Guide

This document lists all debug print statements added to the FaceJudge app.
Use this to understand the flow and identify where problems occur.

Each debug statement is prefixed with [DEBUG:CATEGORY] for easy filtering.

================================
CATEGORIES & FLOW
================================

[DEBUG:APP] - Main application startup/shutdown

- App initialization
- Window creation
- Component loading
- Shutdown sequence

[DEBUG:PROFILE] - Profile panel interactions

- Input field edits
- Save button clicks
- Scraping callbacks

[DEBUG:CACHE] - Local profile caching system

- Cache loads (entries)
- Cache saves (entries)
- Cache hits/misses
- Key-value operations

[DEBUG:RESUME] - Resume file handling

- File extraction attempts
- File type detection (txt/pdf/docx)
- Extraction success/failure

[DEBUG:SCRAPE] - Background profile scraping

- Scrape start
- GitHub fetch (success/fail)
- LinkedIn fetch (success/fail)
- Resume extraction
- Cache persistence
- Background thread management

[DEBUG:RECORD] - Screen recording

- Recording start/stop
- Video file path
- Video codec parameters (resolution, FPS)
- VideoWriter open/close
- Frame write errors

[DEBUG:PAYLOAD] - AI payload construction

- Payload assembly
- Cache lookups for each profile type
- Image and face state inclusion

[DEBUG:AI] - AI generation & TTS

- System prompt selection
- Data inclusion (GitHub/LinkedIn/resume/face)
- Prompt assembly size
- Gemini LLM call
- Response reception
- ElevenLabs TTS setup
- Audio playback
- Errors during any step

[DEBUG:AI_TRIGGER] - Tab key AI trigger sequence

- Trigger start
- Screenshot capture success/fail
- Face state collection
- Profile data assembly
- Payload building
- Background thread spawning
- Mode selection (glaze/hate/super_hate)
- Response reception and display

================================
TYPICAL FLOW
================================

1. APP START:
   [DEBUG:APP] ===== FaceJudge App Starting =====
   [DEBUG:APP] Initializing MainWindow
   [DEBUG:APP] Loading FaceAnalyzer...
   [DEBUG:APP] Loading GestureDetector...
   [DEBUG:APP] Opening camera (cv2.VideoCapture(0))
   [DEBUG:APP] All components initialized successfully
   [DEBUG:APP] Showing fullscreen window
   [DEBUG:APP] ===== FaceJudge Ready =====

2. USER PRESSES SAVE PROFILE:
   [DEBUG:PROFILE] SAVE button clicked
   [DEBUG:PROFILE] Profile data: github=True, linkedin=False, resume_path=True, resume_text=0 chars
   [DEBUG:PROFILE] Scheduling background scrape
   [DEBUG:SCRAPE] Scheduling background scrape in daemon thread
   [DEBUG:SCRAPE] Background thread started
3. BACKGROUND SCRAPE RUNS:
   [DEBUG:SCRAPE] Starting scrape: github=user, linkedin=, resume_path=/path/file.pdf
   [DEBUG:SCRAPE] Fetching GitHub profile for user...
   [DEBUG:SCRAPE] GitHub fetch SUCCESS
   [DEBUG:SCRAPE] Caching GitHub profile for user
   [DEBUG:CACHE] Setting cache key: github::user
   [DEBUG:CACHE] Saved cache: 1 entries to profiles_cache.json
   [DEBUG:CACHE] Successfully cached github::user
   [DEBUG:RESUME] Extracting text from: /path/file.pdf
   [DEBUG:RESUME] File extension: pdf
   [DEBUG:RESUME] Extracting PDF with PyPDF2...
   [DEBUG:RESUME] PDF extracted: 2500 chars
   [DEBUG:SCRAPE] Scrape complete. Result: [...]
   [DEBUG:PROFILE] Background scrape callback fired
   [DEBUG:PROFILE] Scraping complete, marking as saved

4. USER PRESSES TAB (AI TRIGGER):
   [DEBUG:APP] Tab key pressed - triggering AI call
   [DEBUG:AI_TRIGGER] Starting AI trigger sequence
   [DEBUG:AI_TRIGGER] Screenshot captured: 8234 chars (base64)
   [DEBUG:AI_TRIGGER] Face state: emotion=happy, mode=glaze
   [DEBUG:AI_TRIGGER] Profile data collected: gh=True, li=False, resume_path=True, resume_text=0 chars
   [DEBUG:AI_TRIGGER] Building payload with keys: gh_key=github::user, li_key=None, resume_key=resume::/path/file.pdf
   [DEBUG:PAYLOAD] Building payload: image=True, github_key=github::user, linkedin_key=None, resume_key=resume::/path/file.pdf
   [DEBUG:CACHE] Cache HIT: github::user
   [DEBUG:CACHE] Cache MISS: resume::/path/file.pdf
   [DEBUG:PAYLOAD] Payload built: keys=['image_b64', 'face_state', 'github', 'linkedin', 'resume']
   [DEBUG:AI_TRIGGER] Calling AI in background thread with mode=glaze
   [DEBUG:AI_TRIGGER] Background thread started
5. AI GENERATION IN BACKGROUND:
   [DEBUG:AI] Starting AI call with mode: glaze
   [DEBUG:AI] System prompt selected (1200 chars)
   [DEBUG:AI] Including GitHub data
   [DEBUG:AI] Including face analysis
   [DEBUG:AI] Prompt assembled (4500 chars)
   [DEBUG:AI] Calling Gemini LLM...
   [DEBUG:AI] LLM response received (250 chars)
   [DEBUG:AI] ElevenLabs API key found, calling TTS...
   [DEBUG:AI] TTS audio generated, playing...
   [DEBUG:AI] Audio playback started
   [DEBUG:AI] AI call complete

6. USER PRESSES R (RECORDING):
   [DEBUG:APP] R key pressed - recording now STARTING
   [DEBUG:RECORD] Starting screen recording
   [DEBUG:RECORD] Recording to: captured-videos/capture_1715169847.mp4
   [DEBUG:RECORD] Video params: 1280x720 @ 30 fps
   [DEBUG:RECORD] VideoWriter opened successfully

   ... recording continues with frame writes ...

   [DEBUG:APP] R key pressed - recording now STOPPING
   [DEBUG:RECORD] Stopping screen recording
   [DEBUG:RECORD] VideoWriter released

7. APP SHUTDOWN:
   [DEBUG:APP] Quit key pressed (Esc/Q)
   [DEBUG:APP] ===== Shutting Down =====
   [DEBUG:APP] Stopping timer...
   [DEBUG:APP] Releasing camera...
   [DEBUG:APP] Closing analyzer...
   [DEBUG:APP] Closing gesture detector...
   [DEBUG:APP] FaceJudge shutdown complete

================================
HOW TO USE THESE DEBUG STATEMENTS
================================

# Filter for errors only:

python app.py 2>&1 | grep "ERROR\|FAILED\|failed\|WARNING"

# Watch profile caching:

python app.py 2>&1 | grep "DEBUG:CACHE\|DEBUG:PROFILE"

# Watch AI flow:

python app.py 2>&1 | grep "DEBUG:AI"

# Watch all recording events:

python app.py 2>&1 | grep "DEBUG:RECORD"

# Full debug output to file:

python app.py > debug.log 2>&1

# Realtime debug (Windows PowerShell):

python app.py 2>&1 | ForEach-Object { Write-Host $_; if ($\_ -match "ERROR|FAILED") { Write-Host "^^^ ERROR ABOVE ^^^" -ForegroundColor Red } }

================================
COMMON ISSUES & FIXES
================================

"[DEBUG:AI_TRIGGER] WARNING: No face detected"
→ Camera isn't detecting face, or no one is in frame
→ Check lighting, face angle, camera focus

"[DEBUG:CACHE] Cache MISS: github::username"
→ Profile hasn't been scraped yet
→ Click SAVE in profile panel first

"[DEBUG:RESUME] File extraction FAILED"
→ Resume file may be corrupted
→ Try pasting text directly instead

"[DEBUG:AI] LLM call FAILED"
→ Check GEMINI_API_KEY in .env
→ Check internet connection

"[DEBUG:AI] No ElevenLabs API key"
→ ElevenLabs TTS won't work
→ Add ELEVENLABS_API_KEY to .env

"[DEBUG:RECORD] VideoWriter failed to open"
→ Codec not supported on system
→ Check Python OpenCV version

================================
"""

if **name** == "**main**":
print(**doc**)
