# Manim Forge for students

Type a maths or science question and get a short narrated animation that
explains it, made by an AI model running on your own Mac. Nothing is sent to
the internet once it is installed.

## What you need

- A Mac with Apple Silicon (M1 or newer), ideally 16 GB of memory.
- About 12 GB of free disk space.
- An internet connection for the first install (about 7 GB to download).

## Install (once, about 15–30 minutes)

1. Download Manim Forge: on its GitHub page press **Code → Download ZIP** and
   unzip it, or, if you use git: `git clone https://github.com/nimb-ou/manim-forge`.
2. Open **Terminal** (press ⌘-Space, type *Terminal*, press Return).
3. Type `cd ` (with a space), drag the Manim Forge folder into the Terminal
   window, and press Return.
4. Type `bash install.sh` and press Return. It may ask for your Mac password
   (nothing appears as you type; that is normal). When it says **Done**, you
   are ready.

## Use it

Double-click **Start Manim Forge** in the folder. Your browser opens the app;
keep the small Terminal window open while you use it, and close it when you
are finished.

> If macOS says the file "cannot be opened because it is from an
> unidentified developer", right-click it, choose **Open**, then **Open**
> again. You only need to do this once.

Type a question and press **Make the animation**. The first question loads
the AI model (about 20 seconds); after that each animation takes one to three
minutes.

## Asking good questions

It works best on one clear school-level question at a time:

- *a jacket costs £80 and is 15% off. what do you pay?*
- *why is the area of a circle pi r squared?*
- *how do you solve 2x + 5 = 17?*
- *what does the slope of a distance-time graph tell you?*

Give the numbers if you have them. Very long, multi-part, or university-level
questions come out less well.

## Can I trust it?

Read the box next to the video:

- **Numbers checked** (green): every number on screen was worked out by a
  calculation on your computer, not guessed by the AI. The working is listed
  underneath. The *explanation* can still be weak, so think it through.
- **Check this one** (yellow): something could not be checked — the reasons
  are listed. Treat the video as a rough sketch and check the maths yourself
  or with your teacher.

The AI is small and it can be wrong. Use the videos to understand an idea,
not as the final word on an answer. If one is confusing, press **Make another
version** for a different take.

## If something goes wrong

- *"The app is not running"*: double-click **Start Manim Forge** again.
- *"This one did not work"*: ask the question a different way.
- Anything else: run `bash install.sh` again; it fixes most problems and only
  adds what is missing.
