###############################################################################
# clipYoutubeSegments.R  –  minimal downloader + clipper
###############################################################################

# ── 1 · user settings -------------------------------------------------------
csvPath   <- "input.csv"      # full or relative path
outputDir <- "clips"          # folder where trimmed MP4s land

# ── 2 · packages ------------------------------------------------------------
for (p in c("readr", "dplyr", "stringr", "purrr", "fs", "glue"))
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, quiet = TRUE)

library(readr); library(dplyr); library(stringr)
library(purrr); library(fs); library(glue)

# ── 3 · helpers -------------------------------------------------------------
toHms <- function(x) {
  if (is.na(x) || x == "") return(NA_character_)
  if (is.numeric(x)) return(sprintf("%02d:%02d:%02d", x %/% 3600, (x %% 3600) %/% 60, x %% 60))
  if (str_detect(x, "^\\d{1,2}:\\d{1,2}$")) return(glue("00:{x}"))
  x
}

downloadVideo <- function(url, id) {
  file <- path(tempdir(), glue("{id}.mp4"))
  if (!file_exists(file))
    system(glue('yt-dlp -f mp4 -o "{file}" "{url}"'), ignore.stdout = TRUE, ignore.stderr = TRUE)
  file
}

trimWithFfmpeg <- function(src, dst, start, end)
  system(glue('ffmpeg -hide_banner -loglevel error -ss {start} -to {end} -i "{src}" -c copy "{dst}"'),
         ignore.stdout = TRUE, ignore.stderr = TRUE)

# ── 4 · data ---------------------------------------------------------------
clips <- read_csv(csvPath, show_col_types = FALSE) %>%
  mutate(
    start   = map_chr(as.character(start), toHms),
    end     = map_chr(as.character(end),   toHms),
    videoId = str_extract(url, "(?<=v=|be/)[A-Za-z0-9_-]{11}"),
    idx     = row_number()
  ) %>%
  filter(!is.na(start) & !is.na(end))

if (nrow(clips) == 0) stop("No usable rows in CSV.")

dir_create(outputDir)

# ── 5 · main loop -----------------------------------------------------------
pwalk(
  clips,
  function(url, start, end, videoId, idx, ...) {
    message(glue("[{idx}/{nrow(clips)}] {url}"))
    src <- downloadVideo(url, videoId)
    dst <- path(
      outputDir,
      glue("{sprintf('%02d', idx)}_{videoId}_{str_replace_all(start, ':', '-')}-{str_replace_all(end, ':', '-')}.mp4")
    )
    trimWithFfmpeg(src, dst, start, end)
    message("    → saved ", dst)
  }
)

message(glue("\nAll done. Clips in: {path_abs(outputDir)}"))
