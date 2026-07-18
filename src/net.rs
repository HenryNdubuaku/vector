use std::fs;
use std::process::Command;

use crate::runtime::fnv64;
use crate::{die, home};

pub fn is_url(path: &str) -> bool {
    path.starts_with("http://") || path.starts_with("https://")
}

pub fn fetch(url: &str) -> String {
    let bare = url.split(['?', '#']).next().unwrap();
    let ext = bare.rsplit('/').next().unwrap().rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if !matches!(ext, "npy" | "csv" | "png" | "wav" | "safetensors" | "txt" | "json" | "gz" | "jpg" | "jpeg" | "mp3" | "flac" | "ogg") {
        die(&format!("cannot tell the format of {}; the url must end in .npy, .csv, .png, .wav, .txt, .json or .safetensors", url));
    }
    let dir = format!("{}/.vector/downloads", home());
    let path = format!("{}/{:016x}.{}", dir, fnv64(url.as_bytes()), ext);
    if fs::metadata(&path).is_ok() {
        return path;
    }
    fs::create_dir_all(&dir).unwrap_or_else(|e| die(&format!("cannot create {}: {}", dir, e)));
    let tmp = format!("{}.part", path);
    eprintln!("downloading {}", url);
    let status = Command::new("curl")
        .args(["-fL", "--progress-bar", "-o", &tmp, url])
        .status()
        .unwrap_or_else(|e| die(&format!("cannot run curl: {}", e)));
    if !status.success() {
        let _ = fs::remove_file(&tmp);
        die(&format!("cannot download {}", url));
    }
    fs::rename(&tmp, &path).unwrap_or_else(|e| die(&format!("cannot write {}: {}", path, e)));
    path
}
