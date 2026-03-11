from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8503")
    time.sleep(2)
    # Click on the center of the viewport or find a specific element
    page.mouse.click(400, 300)
    time.sleep(1)
    page.mouse.click(400, 400)
    time.sleep(2)
    content = page.content()
    if 'event' in content.lower() or 'selection' in content.lower() or 'points' in content.lower():
        print(content)
    else:
        print("NO EVENT RENDERED")
    browser.close()
