# FaceJudge - Enhanced Version

A PyQt6 + OpenCV desktop application with AI-powered roasting and glazing, powered by Gemini LLM and ElevenLabs TTS.

## ✨ New Features

### 1. **Profile Scraping & Caching**

- Automatically scrapes GitHub and LinkedIn profiles in the background
- Caches all scraped data locally in `profiles_cache.json`
- Only re-scrapes if profile URLs change (smart caching)
- Supports resume upload (PDF, DOCX, TXT) or inline text input

### 2. **AI-Powered Roasting/Glazing**

Press **Tab** to trigger the main AI call with:

- Live screenshot (Base64-encoded face crop)
- OpenCV face metrics (FWHR, jaw ratio, symmetry, etc.)
- Cached LinkedIn & GitHub profile data
- Resume text
- Detected facial emotion

Three modes available:

- **Glaze Mode** (👍 up gesture): Hyperbolic compliments
- **Hate Mode** (👎 down gesture): Savage roasts
- **Super Hate Mode** (two 👎👎): Maximum unhinged chaos

### 3. **Screen Recording**

Press **r** to toggle screen recording.  
Videos are saved to `captured-videos/` folder with timestamp.  
Red indicator dot appears in top-left when recording.

### 4. **Enhanced UI**

- Profile panel with resume text input field
- "SAVING..." → "SAVED ✓" checkmark flow
- Checkmark resets when inputs are edited
- Updated hint text showing all keybinds

### 5. **Audio Output**

AI responses are automatically passed to ElevenLabs TTS and played aloud.

## 🚀 Quick Start

### Prerequisites

```bash
# All required packages:
pip install PyQt6 opencv-python mediapipe numpy requests elevenlabs python-docx PyPDF2
```

### Setup Environment

Create `.env` file in project root:

```
GEMINI_API_KEY=your_key_here
# OR
GOOGLE_API_KEY=your_key_here

ELEVENLABS_API_KEY=your_key_here
```

### Run the App

```bash
python app.py
```

## ⌨️ Keybinds

| Key       | Action                                                |
| --------- | ----------------------------------------------------- |
| **Tab**   | Trigger AI (roast/glaze/super-hate based on mode)     |
| **r**     | Toggle screen recording (saves to `captured-videos/`) |
| **Space** | Capture screenshot of current face                    |
| **m**     | Toggle face mesh visualization                        |
| **Click** | Cycle to next detected face                           |
| **Esc**   | Quit                                                  |

## 📁 File Structure

```
.
├── app.py                    # Main PyQt6 GUI & video loop
├── ai_core.py               # New: AI orchestration, caching, scraping
├── cvModule.py              # Face analysis & metrics
├── mainAI.py                # Gemini LLM integration (refactored)
├── githubinfo.py            # GitHub profile scraper
├── linkedin_profile.py       # LinkedIn profile scraper
├── profiles_cache.json       # Auto-generated: cached profiles
├── captured-videos/         # Auto-created: screen recording videos
├── captures/                # Face capture screenshots
├── templates/index.html     # (Legacy Flask template)
└── .env                      # API keys (not committed)
```

## 🔧 How It Works

### Profile Scraping Flow

1. User inputs GitHub username, LinkedIn username, resume (file or text)
2. Click **SAVE** → background thread starts scraping
3. Data cached locally with unique keys (`github::username`, `linkedin::username`, etc.)
4. "SAVED ✓" checkmark appears when done
5. Checkmark resets if user edits any field

### AI Generation Flow (Tab Key)

1. Screenshot captured & encoded to Base64
2. Face metrics extracted (emotion, FWHR, symmetry, etc.)
3. Cached profiles looked up from `profiles_cache.json`
4. Payload constructed with all data
5. Gemini LLM called with appropriate system prompt (glaze/hate/super_hate)
6. Response text passed to ElevenLabs TTS
7. Audio plays automatically
8. Response printed to console

### Gesture Recognition

- **Thumb Side** → Neutral (no action)
- **Thumb Up** → Glaze Mode (compliments)
- **Thumb Down** → Hate Mode (roasts)
- **Two Thumbs Down** → Super Hate Mode (unhinged)

## 📊 System Prompts

### Glaze Mode

Creative, poetic, over-the-top compliments combining facial features with real achievements.

### Hate Mode

Savage but clever roasts merging facial features with online footprint.

### Super Hate Mode

Maximum unhinged chaos - deranged, creative, hilariously brutal.

## 🛠️ Architecture Notes

**ai_core.py** (new core module):

- `schedule_background_scrape()` - runs scraping in daemon thread
- `build_payload()` - assembles API request with cached data
- `call_ai_and_speak()` - calls Gemini + ElevenLabs
- Caching system with file locks for thread safety

**mainAI.py** refactored:

- Wrapped interactive code in `if __name__ == "__main__":` block
- Made module importable without prompts

## 🐛 Troubleshooting

**"SAVED ✓ not appearing?"**

- Check `.env` has valid API keys
- See browser console for scraping errors (printed to stdout)

**"No audio?"**

- Verify `ELEVENLABS_API_KEY` is set
- Check system volume

**"Recording not working?"**

- Ensure `captured-videos/` folder is writable
- Check OpenCV VideoWriter support on your system

**"Import errors?"**

- Activate venv: `.\.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (Unix)
- Install all deps: `pip install -r requirements.txt` (if exists)

## 📝 Future Enhancements

- [ ] Cache expiration (refresh data after N days)
- [ ] Confidence scores for scraped data
- [ ] Multiple voice selections for TTS
- [ ] Custom system prompts via UI
- [ ] Video preview before upload
- [ ] Analytics dashboard

## 📄 License

(Add your license here)
