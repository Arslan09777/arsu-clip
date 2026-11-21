# Arsu Clip

Arsu Clip turns a YouTube video or a local video file into captioned clips. Run it on your computer, review the results in the Clip Library, and publish the clips you choose.

Maintained by [Arslan09777](https://github.com/Arslan09777). Based on [Clipsmith](https://github.com/AbdulSamad94/clipsmith) by AbdulSamad94 and the Clipsmith contributors. See the [MIT license](LICENSE) and [font licenses](fonts/OFL.txt).

## Install and run

You need Git, Python 3.10 or newer, and FFmpeg (`ffmpeg` and `ffprobe` on your PATH). Install Deno as well if you plan to import from YouTube. Restart your terminal after installing prerequisites.

```sh
git clone https://github.com/Arslan09777/arsu-clip.git
cd arsu-clip
```

| Platform | Install dependencies | Start the studio |
| --- | --- | --- |
| Windows | Run `setup.bat`. Install FFmpeg with `winget install Gyan.FFmpeg` and Deno with `winget install DenoLand.Deno`. | Run `run.bat`. |
| macOS | Install FFmpeg and Deno with `brew install ffmpeg deno`, then run `sh setup.sh`. | Run `sh run.sh`. |
| Linux | Install FFmpeg, Python with venv support, and [Deno](https://deno.com/), then run `sh setup.sh`. | Run `sh run.sh`. |

Open [Arsu Clip on localhost](http://127.0.0.1:8765). Keep the terminal running while generating or uploading clips. The launcher uses port 8765. If you already run the studio on another port or keep your library in a custom `CLIPSMITH_DATA_DIR`, keep those settings to open the same library.

## Create clips

1. Paste a YouTube link or upload a local video in **Create**.
2. Click **Queue project**. **Clip length**, **Number of clips**, and **Framing** default to **Auto**. Auto length uses a 15–90 second range. Auto count aims for 10–30 clips according to the material available. Open **Clip settings** to override them.
3. The default output is vertical **9:16**, with yellow Hinglish captions, opening hooks, and pause trimming. Choose a caption language, style, or aspect ratio if you need a different output.
4. Watch **Processing** for progress. Preview or download MP4s from the **Clip Library**. Review generated titles, captions, and hashtags before posting.

The fresh install uses basic scoring without a model or API key. For model-based selection, open **Integrations → AI provider**, configure a provider such as Gemini or an OpenAI-compatible endpoint, test it, and save it as the default. Ollama is optional. You can also set up providers in `.env` using [the example file](.env.example). If a source has no usable YouTube captions, local Whisper can transcribe it and may download a model on first use. Configured cloud providers may receive transcript text or audio for the features you enable.

## Connect publishing

| Platform | Setup | Posting |
| --- | --- | --- |
| YouTube | [Google OAuth setup](docs/youtube-setup.md) | Post from a clip card, or opt in to auto-post or a schedule when creating a project. Keep Arsu Clip running until YouTube accepts the upload. |
| Instagram | [Meta account setup](docs/meta-setup.md) | Post a Reel from a clip card, or opt in to auto-post or a schedule. Arsu Clip must be running when a scheduled upload is due. |
| Facebook | [Meta account setup](docs/meta-setup.md) | Post a Reel directly from a clip card using the connected Facebook Page. |

Auto-posting is opt-in. For scheduled projects, set a first posting time and the gap between clips. Confirm the time zone displayed in the form. Review the platform-specific publish result in Arsu Clip and use the link it supplies to check the post.

## Local files and troubleshooting

- Keep `.env`, OAuth credentials, access tokens, source media, rendered clips, and the `data/` directory on your own machine. The repository includes a blank configuration example.
- If FFmpeg is missing, install it and reopen your terminal. If a YouTube import fails, check Deno, update `yt-dlp` in the virtual environment, or use a local video file.
- If a platform rejects a post, read the clip's status before retrying. If publication is unconfirmed, check the destination account for the Reel before trying again.
- The app listens on localhost and has no user login. Keep it on your own computer.

## Development

Install `requirements-dev.txt` inside the project virtual environment and run `python -m pytest` from the repository root.
