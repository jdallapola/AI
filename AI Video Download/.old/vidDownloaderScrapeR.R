library(chromote)
library(glue)
library(stringr)
library(tibble)
library(dplyr)
library(fs)

videoUrl <- "https://www.youtube.com/watch?v=6_m6a0zeLq0"

b <- ChromoteSession$new()  # launches headless Chrome

b$view()                    # <‑‑ opens a real browser window you can see

# now drive the page; everything you script will be mirrored in that window
b$Page$navigate("https://www.youtube-transcript.io/")

# --- paste URL and trigger events ------------------------------------------
jsFillAndSubmit <- sprintf('
const url = "%s";
const inp = document.querySelector("input[type=\\"url\\"]");

// ---- Step 1 · set value via native setter so React sees it ----
const nativeSetter = Object.getOwnPropertyDescriptor(
  window.HTMLInputElement.prototype, "value"
).set;
nativeSetter.call(inp, url);

// ---- Step 2 · fire input event ----
inp.dispatchEvent(new Event("input", { bubbles: true, composed: true }));

// ---- Step 3 · after one tick, click the submit button ----
setTimeout(() => {
  document.querySelector("form button[type=\\"submit\\"]").click();
}, 100);
', videoUrl)

b$Runtime$evaluate(jsFillAndSubmit)

jsDownload <- '
(async () => {

  /* 1 · click the ellipsis button */
  const ellipsis = document
      .querySelector("button svg.lucide-ellipsis-vertical")?.parentElement;
  if (!ellipsis) throw "ellipsis button not found";
  ellipsis.click();

  /* 2 · wait up to 4 s for a menuitem that contains the download icon */
  const wait = ms => new Promise(r => setTimeout(r, ms));
  let item = null;
  for (let i = 0; i < 80; i++) {            // 80 × 50 ms = 4 s
    await wait(50);
    item = [...document.querySelectorAll("svg.lucide-download")]
             .map(svg => svg.closest("[role=\\"menuitem\\"]"))
             .find(Boolean);
    if (item) break;
  }
  if (!item) throw "Download menu item not found";
  item.click();                             // 3 · click it

})();'

b$Runtime$evaluate(jsDownload, awaitPromise = TRUE)




jsClickDT <- '
(async () => {
  // wait up to 2 s for the modal that contains the Download‑Transcript button
  const xpath = "//*[@id=\\"radix-:r8:\\"]/div[3]/button/button";
  const timeout = 2000;                    // ms
  const t0 = Date.now();
  let btn = null;

  while (Date.now() - t0 < timeout) {
    btn = document.evaluate(xpath, document, null,
             XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    if (btn) break;
    await new Promise(r => setTimeout(r, 50));
  }
  if (!btn) throw "Download‑Transcript button not found";

  btn.click();                             // fire!
})();
'

b$Runtime$evaluate(jsClickDT, awaitPromise = TRUE)



# ------------------------------------------------------------------
# the folder Chrome writes to (set this in Browser.setDownloadBehavior)
dlDir <- tempdir()                      #  e.g. "C:\\Users\\...\\AppData\\Local\\Temp\\Rtmp..."

# ------------------------------------------------------------------
# locate the newest .txt file in that folder
file <- fs::dir_ls(dlDir, glob = "*.txt", recurse = FALSE) |>
  fs::file_info() |>
  arrange(desc(modification_time)) |>
  slice(1) |>
  pull(path)

stopifnot(length(file) == 1)            # bail if no file found

# ------------------------------------------------------------------
# read + parse timestamps
lines <- readLines(file, warn = FALSE, encoding = "UTF-8")

transcript <- tibble(raw = lines) |>
  filter(str_detect(raw, "^\\d{2}:\\d{2}:\\d{2}")) |>
  transmute(
    start = str_extract(raw, "^\\d{2}:\\d{2}:\\d{2}"),
    text  = str_trim(str_remove(raw, "^\\d{2}:\\d{2}:\\d{2}\\s+"))
  )

print(head(transcript, 10))

# ------------------------------------------------------------------
# optional: delete the raw txt
fs::file_delete(file)



















