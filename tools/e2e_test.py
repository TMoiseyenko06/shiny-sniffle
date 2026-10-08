"""Headless browser test of the GUI at phone width: login, upload, progress, playback, download, errors.

    python tools/e2e_test.py --password '<GUI_PASSWORD>' [--url http://127.0.0.1:8000]
                             [--image photo.jpg] [--duration 5] [--resolution 480p] [--timeout 1800]
                             [--mock-failure]   (mock backend only: also checks the failed-job card)

Needs Playwright:  pip install playwright && playwright install chromium
Screenshots go to ./e2e-screenshots/. Exit code is non-zero if any check fails.
"""
import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

PHONE = dict(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True,
             user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 "
                        "(KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  - ' + str(detail) if detail else ''}", flush=True)


def no_horizontal_scroll(page, label):
    width = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    check(f"no horizontal scroll ({label})", width[0] <= width[1], f"scrollWidth={width[0]} viewport={width[1]}")


def small_tap_targets(page):
    """Visible interactive elements shorter than 44 px."""
    return page.evaluate("""() => [...document.querySelectorAll(
        'button, a.btn, select, input:not([type=file]):not([type=radio]):not([type=checkbox]), textarea, summary, .segmented label, label.switch, .picker')]
      .filter(e => e.offsetParent !== null)
      .map(e => [e.id || e.className || e.tagName, Math.round(e.getBoundingClientRect().height)])
      .filter(([, h]) => h < 44)""")


def default_image(directory):
    from PIL import Image, ImageDraw
    path = Path(directory) / "test-photo.jpg"
    img = Image.new("RGB", (1200, 1600), (40, 70, 120))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 1000, 1200, 1600), fill=(60, 120, 60))
    draw.ellipse((420, 300, 780, 660), fill=(250, 210, 60))
    img.save(path, quality=92)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--password", required=True)
    parser.add_argument("--image")
    parser.add_argument("--prompt", default="The camera slowly pushes in while the sun rises and the clouds drift.")
    parser.add_argument("--duration", default="5")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds to wait for the generation")
    parser.add_argument("--mock-failure", action="store_true")
    parser.add_argument("--screenshots", default="e2e-screenshots")
    parser.add_argument("--chromium", help="path to a Chromium/Chrome binary")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    shots = Path(args.screenshots)
    shots.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    image = args.image or default_image(tmp)
    big = Path(tmp) / "too-big.jpg"
    big.write_bytes(b"\xff\xd8\xff" + b"0" * (26 * 1024 * 1024))

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
        context = browser.new_context(accept_downloads=True, **PHONE)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)

        # --- login
        page.goto(url + "/")
        check("unauthenticated visit redirects to /login", page.url.endswith("/login"), page.url)
        page.fill("#password", "definitely-wrong")
        page.click("#go")
        expect(page.locator("#err")).to_be_visible()
        check("wrong password shows an error", "Wrong password" in page.locator("#err").inner_text())
        page.screenshot(path=shots / "01-login.png")
        page.fill("#password", args.password)
        page.click("#go")
        page.wait_for_url(url + "/")
        check("correct password opens the app", page.url == url + "/")
        expect(page.locator("#status-model")).to_have_text(re.compile("Model ready|Loading model"), timeout=15000)
        check("status bar shows model state and GPU", True, page.locator("#status").inner_text().replace("\n", " "))
        no_horizontal_scroll(page, "empty page")
        page.screenshot(path=shots / "02-empty.png", full_page=True)

        # --- failure: oversized file (rejected in the browser before uploading)
        page.set_input_files("#image", str(big))
        expect(page.locator("#form-error")).to_be_visible()
        check("oversized photo is rejected with a clear message", "limit is 25 MB" in page.locator("#form-error").inner_text(),
              page.locator("#form-error").inner_text())
        # --- failure: not really an image (rejected by the server)
        page.set_input_files("#image", files=[{"name": "fake.jpg", "mimeType": "image/jpeg", "buffer": b"not an image"}])
        page.fill("#prompt", "test")
        page.click("#submit")
        expect(page.locator("#form-error")).to_contain_text("not a readable image", timeout=10000)
        check("server-side validation error is shown in the form", True, page.locator("#form-error").inner_text())
        page.screenshot(path=shots / "03-upload-error.png", full_page=True)

        # --- real job
        page.set_input_files("#image", image)
        expect(page.locator("#preview")).to_be_visible()
        check("thumbnail preview appears", page.locator("#preview").is_visible())
        page.fill("#prompt", args.prompt)
        page.click("#advanced summary")
        page.locator("#duration").fill(str(args.duration))
        page.select_option("#resolution", args.resolution)
        small = small_tap_targets(page)
        check("all tap targets are at least 44 px tall", not small, small)
        no_horizontal_scroll(page, "form with Advanced open")
        page.screenshot(path=shots / "04-form.png", full_page=True)

        jobs_before = page.locator(".job").count()
        page.click("#submit")
        expect(page.locator(".job")).to_have_count(jobs_before + 1, timeout=60000)
        expect(page.locator("#submit")).to_be_enabled(timeout=60000)
        check("Generate re-enables after submitting and a card appears", True)
        card = page.locator(".job").first
        expect(card.locator(".badge")).to_have_text(re.compile("Queued|Running|Done"))

        saw_segment, saw_progress = False, False
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            status = card.locator(".badge").inner_text().lower()
            if status == "running":
                label = card.locator(".progress-label").inner_text()
                width = card.locator(".bar > span").evaluate("e => parseFloat(e.style.width) || 0")
                if "Segment" in label and not saw_segment:
                    saw_segment = True
                    page.screenshot(path=shots / "05-progress.png", full_page=True)
                    check("running card shows segment progress", True, f"{label} | {card.locator('.elapsed').inner_text()}")
                saw_progress = saw_progress or width > 0
            if status in ("done", "failed"):
                break
            time.sleep(0.5)
        status = card.locator(".badge").inner_text().lower()
        check("progress bar advanced", saw_progress)
        check("job finished", status == "done", status if status != "failed" else card.locator(".job-error").inner_text())
        if status == "done":
            video = card.locator("video")
            expect(video).to_be_visible()
            attrs = video.evaluate("v => [v.hasAttribute('playsinline'), v.hasAttribute('controls'), v.getAttribute('src')]")
            check("finished card has an inline player (playsinline, controls)", attrs[0] and attrs[1], attrs[2])
            h264 = page.evaluate("document.createElement('video').canPlayType('video/mp4; codecs=\"avc1.42E01E\"')")
            if h264:
                loaded = video.evaluate("""v => new Promise(r => { if (v.readyState >= 1) return r(v.duration);
                    v.onloadedmetadata = () => r(v.duration); v.onerror = () => r('error ' + (v.error && v.error.code));
                    setTimeout(() => r('timeout'), 15000); })""")
                check("video loads in the browser", isinstance(loaded, (int, float)) and loaded > 0, f"duration={loaded}")
            else:
                print("SKIP  in-browser playback: this Chromium build has no H.264 decoder (Chrome/Safari do)")
            loop = card.locator("button", has_text="Loop")
            loop.click()
            check("Loop toggle works", loop.get_attribute("aria-pressed") == "true" and video.evaluate("v => v.loop"))
            with page.expect_download() as info:
                card.locator("a", has_text="Download").click()
            path = info.value.path()
            head = Path(path).read_bytes()[:12]
            check("Download delivers an MP4", head[4:8] == b"ftyp", f"{Path(path).stat().st_size} bytes, {info.value.suggested_filename}")
            no_horizontal_scroll(page, "finished card")
            card.scroll_into_view_if_needed()
            page.screenshot(path=shots / "06-done.png", full_page=True)

            card.locator("button", has_text="Reuse").click()
            expect(page.locator("#prompt")).to_have_value(args.prompt)
            check("Reuse settings refills the form", page.locator("#image-meta").inner_text().startswith("Using the photo"))

        # --- page refresh keeps state
        count = page.locator(".job").count()
        page.reload()
        expect(page.locator(".job")).to_have_count(count, timeout=15000)
        check("jobs survive a page refresh", True, f"{count} cards")

        # --- failed card + retry + delete (mock backend can fake a CUDA OOM)
        if args.mock_failure:
            page.set_input_files("#image", image)
            page.fill("#prompt", "simulate OOM: this one should fail")
            page.click("#submit")
            failed = page.locator(".job").first
            expect(failed.locator(".badge")).to_have_text("Failed", timeout=120000)
            error = failed.locator(".job-error").inner_text()
            check("failed card shows a readable error", "ran out of memory" in error, error)
            check("failed card has Retry", failed.locator("button", has_text="Retry").is_visible())
            failed.scroll_into_view_if_needed()
            page.screenshot(path=shots / "07-failed.png", full_page=True)
            failed.locator("button", has_text="Retry").click()
            expect(failed.locator(".badge")).to_have_text(re.compile("Queued|Running"), timeout=10000)
            check("Retry re-queues the job", True)
            page.once("dialog", lambda d: d.accept())
            before = page.locator(".job").count()
            failed.locator("button", has_text=re.compile("Cancel|Delete")).click()
            expect(page.locator(".job")).to_have_count(before - 1, timeout=10000)
            check("Cancel/Delete with confirm removes the card", True)

        check("no JavaScript errors", not errors, errors[:3])

        desktop = browser.new_context(viewport={"width": 1280, "height": 900})
        desktop.add_cookies(context.cookies())
        dpage = desktop.new_page()
        dpage.goto(url + "/")
        expect(dpage.locator("#status-model")).to_have_text(re.compile("Model|Loading"), timeout=15000)
        dpage.wait_for_timeout(1000)
        dpage.screenshot(path=shots / "08-desktop.png")
        browser.close()

    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed. Screenshots in {shots}/")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
