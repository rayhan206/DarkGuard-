import asyncio
import ipaddress
import socket
from urllib.parse import urlparse


VIEWPORT = {"width": 1365, "height": 768}
DEMO_PATHS = {
    "/demo/safe", "/demo/urgency", "/demo/scarcity", "/demo/social_proof",
    "/demo/confirmshaming", "/demo/preselection", "/demo/visual_interference",
    "/demo/dynamic_timer",
}


def validate_url(url: str, allow_local_demo: bool = False) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs can be analysed.")
    if not parsed.hostname:
        raise ValueError("The URL must include a valid host name.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing user names or passwords are not allowed.")

    is_demo = (
        allow_local_demo
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        and parsed.path in DEMO_PATHS
    )
    if is_demo:
        return url

    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or default_port)
    except socket.gaierror as error:
        raise ValueError("The URL host could not be resolved.") from error
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private-network and localhost URLs are not allowed.")
    return url


OBSERVER_SCRIPT = """
() => {
  const markAdded = node => {
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    node.__dgAppeared = true;
    node.querySelectorAll('*').forEach(child => child.__dgAppeared = true);
  };
  window.__darkguardObserver = new MutationObserver(records => {
    records.forEach(record => {
      if (record.type === 'childList') {
        record.addedNodes.forEach(markAdded);
        let parent = record.target.nodeType === Node.TEXT_NODE ? record.target.parentElement : record.target;
        if (parent && record.removedNodes.length && record.addedNodes.length) {
          parent.__dgTextChangeCount = (parent.__dgTextChangeCount || 0) + 1;
        }
      }
      if (record.type === 'characterData') {
        const parent = record.target.parentElement;
        if (parent) parent.__dgTextChangeCount = (parent.__dgTextChangeCount || 0) + 1;
      }
      if (record.type === 'attributes') {
        record.target.__dgAttributeChangeCount = (record.target.__dgAttributeChangeCount || 0) + 1;
      }
    });
  });
  window.__darkguardObserver.observe(document.documentElement, {
    subtree: true, childList: true, characterData: true, attributes: true,
    attributeFilter: ['checked', 'hidden', 'style', 'class', 'aria-hidden']
  });
}
"""


EXTRACT_SCRIPT = """
() => {
  const selector = 'button,a,input,label,dialog,[role="button"],[role="dialog"],body *';
  const items = [];
  const seen = new Set();
  for (const element of document.querySelectorAll(selector)) {
    if (seen.has(element)) continue;
    seen.add(element);
    const tag = element.tagName.toLowerCase();
    const role = element.getAttribute('role') || '';
    const interactive = ['button','a','input','label','dialog'].includes(tag) || ['button','dialog'].includes(role);
    const hasElementChild = element.children.length > 0;
    const ownText = (element.innerText || element.textContent || '').replace(/\s+/g, ' ').trim();
    if (!interactive && (hasElementChild || !ownText)) continue;

    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    const visible = rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0 &&
      rect.top < innerHeight && rect.left < innerWidth && style.display !== 'none' &&
      style.visibility !== 'hidden' && Number(style.opacity) > 0.02;
    if (!visible) continue;

    let text = ownText;
    if (tag === 'input' && !text) {
      text = Array.from(element.labels || []).map(label => label.innerText).join(' ').replace(/\s+/g, ' ').trim();
      if (!text && element.parentElement?.tagName.toLowerCase() === 'label') text = element.parentElement.innerText.trim();
    }
    if (!text && !['input','dialog'].includes(tag)) continue;
    items.push({
      text: text.slice(0, 500), tag, role, input_type: element.type || '',
      checked: Boolean(element.checked), disabled: Boolean(element.disabled), required: Boolean(element.required),
      bbox: [Math.max(0, rect.left), Math.max(0, rect.top), Math.min(innerWidth, rect.right), Math.min(innerHeight, rect.bottom)],
      font_size: parseFloat(style.fontSize) || 16, color: style.color,
      background_color: style.backgroundColor, opacity: Number(style.opacity),
      display: style.display, visibility: style.visibility, z_index: parseInt(style.zIndex) || 0,
      aria_label: element.getAttribute('aria-label') || '', appeared_after_load: Boolean(element.__dgAppeared),
      text_change_count: element.__dgTextChangeCount || 0,
      attribute_change_count: element.__dgAttributeChangeCount || 0,
      animation_count: element.getAnimations().filter(a => a.playState === 'running').length
    });
  }
  return {elements: items, title: document.title, final_url: location.href,
    active_animation_count: document.getAnimations().filter(a => a.playState === 'running').length};
}
"""


async def capture_page(url: str, observe_seconds: float = 2) -> dict:
    allow_demo = urlparse(url).path in DEMO_PATHS
    validate_url(url, allow_local_demo=allow_demo)
    try:
        from playwright.async_api import async_playwright
    except ImportError as error:
        raise RuntimeError("Playwright is not installed. Run: pip install -r requirements.txt") from error

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport=VIEWPORT,
                device_scale_factor=1,
                ignore_https_errors=False,
            )
            page = await context.new_page()
            page.set_default_timeout(12_000)
            response = await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            if response and response.status >= 400:
                raise RuntimeError(f"The webpage returned HTTP {response.status}.")
            await page.evaluate(OBSERVER_SCRIPT)
            await asyncio.sleep(observe_seconds)
            data = await page.evaluate(EXTRACT_SCRIPT)
            final_url = data["final_url"]
            validate_url(final_url, allow_local_demo=urlparse(final_url).path in DEMO_PATHS)
            screenshot = await page.screenshot(full_page=False, type="png")
            await context.close()
            await browser.close()
    except RuntimeError:
        raise
    except Exception as error:
        message = str(error).splitlines()[0]
        if "Executable doesn't exist" in str(error):
            message = "Chromium is missing. Run: playwright install chromium"
        raise RuntimeError(f"The rendered page could not be captured: {message}") from error

    data["screenshot"] = screenshot
    data["viewport"] = VIEWPORT.copy()
    return data
