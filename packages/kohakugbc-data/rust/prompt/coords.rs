//! Box coordinates as written in a prompt, and the header that declares them
//! (`kohakugbc_data.prompt.coords`).

use std::sync::LazyLock;

use regex::Regex;

use super::dag::Box4;

pub static HEADER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^<gbc( [a-z_]+=[^ >]+)*>$").unwrap());
static FIELD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" ([a-z_]+)=([^ >]+)").unwrap());

#[derive(Clone, Debug, PartialEq)]
pub struct Coords {
    pub width: i64,
    pub height: i64,
    /// `abs_centralized` (pixels from the canvas centre) | `relative` (`[0, scale]`).
    pub space: String,
    /// `yxyx` | `xyxy`.
    pub order: String,
    pub scale: i64,
    /// `enclose` (start edges down, end edges up, after Python's `round(., 9)`) | `nearest`.
    pub rounding: String,
}

impl Default for Coords {
    fn default() -> Self {
        Coords {
            width: 1024,
            height: 1024,
            space: "abs_centralized".into(),
            order: "yxyx".into(),
            scale: 1000,
            rounding: "enclose".into(),
        }
    }
}

impl Coords {
    pub fn new(
        width: i64,
        height: i64,
        space: &str,
        order: &str,
        scale: i64,
        rounding: &str,
    ) -> Result<Self, String> {
        let c = Coords {
            width,
            height,
            space: space.into(),
            order: order.into(),
            scale,
            rounding: rounding.into(),
        };
        if !matches!(space, "abs_centralized" | "relative")
            || !matches!(order, "yxyx" | "xyxy")
            || !matches!(rounding, "enclose" | "nearest")
        {
            return Err(format!(
                "space, order or rounding not in (abs_centralized, relative), (yxyx, xyxy), \
                 (enclose, nearest): {c:?}"
            ));
        }
        if width.min(height).min(scale) <= 0 {
            return Err(format!("width, height and scale are positive: {c:?}"));
        }
        Ok(c)
    }

    /// (size, offset) of x, then of y: written = rounded(stored * size) - offset.
    fn axes(&self) -> ((i64, i64), (i64, i64)) {
        if self.space == "relative" {
            return ((self.scale, 0), (self.scale, 0));
        }
        (
            (self.width, self.width.div_euclid(2)),
            (self.height, self.height.div_euclid(2)),
        )
    }

    pub fn write(&self, b: &Box4) -> [i64; 4] {
        let ((sx, ox), (sy, oy)) = self.axes();
        let (fx, fy) = (sx as f64, sy as f64);
        type Edge = fn(f64) -> i64;
        let (lo, hi): (Edge, Edge) = if self.rounding == "enclose" {
            (|v| round9(v).floor() as i64, |v| round9(v).ceil() as i64)
        } else {
            (
                |v| v.round_ties_even() as i64,
                |v| v.round_ties_even() as i64,
            )
        };
        let (x0, y0) = (lo(b[0] * fx) - ox, lo(b[1] * fy) - oy);
        let (x1, y1) = (hi(b[2] * fx) - ox, hi(b[3] * fy) - oy);
        if self.order == "yxyx" {
            [y0, x0, y1, x1]
        } else {
            [x0, y0, x1, y1]
        }
    }

    pub fn read(&self, ints: [i64; 4]) -> Result<Box4, String> {
        let ((sx, ox), (sy, oy)) = self.axes();
        let [a, b, c, d] = ints;
        let (y0, x0, y1, x1) = if self.order == "yxyx" {
            (a, b, c, d)
        } else {
            (b, a, d, c)
        };
        let at = |v: i64, s: i64, o: i64| -> Result<f64, String> {
            match v.checked_add(o) {
                Some(p) if (0..=s).contains(&p) => Ok(p as f64 / s as f64),
                _ => Err(format!("{ints:?}: off the canvas of {}", self.header())),
            }
        };
        Ok([
            at(x0, sx, ox)?,
            at(y0, sy, oy)?,
            at(x1, sx, ox)?,
            at(y1, sy, oy)?,
        ])
    }

    pub fn snap(&self, b: &Box4) -> Result<Box4, String> {
        self.read(self.write(b))
    }

    pub fn header(&self) -> String {
        let size = if self.space == "abs_centralized" {
            format!(" canvas={}x{}", self.width, self.height)
        } else {
            String::new()
        };
        let scale = if self.space == "relative" {
            format!(" scale={}", self.scale)
        } else {
            String::new()
        };
        format!(
            "<gbc bbox={} space={}{size}{scale}>",
            self.order, self.space
        )
    }

    pub fn from_header(line: &str) -> Result<Self, String> {
        if !HEADER.is_match(line) {
            return Err(format!("not a header: {line:?}"));
        }
        let mut f = std::collections::HashMap::new();
        for c in FIELD.captures_iter(line) {
            f.insert(c[1].to_string(), c[2].to_string());
        }
        let get = |k: &str, d: &str| f.get(k).cloned().unwrap_or_else(|| d.to_string());
        let int = |s: &str| {
            s.parse::<i64>()
                .map_err(|_| format!("invalid literal for int(): {s:?}"))
        };
        let canvas = get("canvas", "1024x1024");
        let parts: Vec<&str> = canvas.split('x').collect();
        if parts.len() != 2 {
            return Err(format!("canvas {canvas:?} is not WxH"));
        }
        Coords::new(
            int(parts[0])?,
            int(parts[1])?,
            &get("space", "abs_centralized"),
            &get("bbox", "yxyx"),
            int(&get("scale", "1000"))?,
            "enclose",
        )
    }
}

/// Python's `round(v, 9)`: the shortest correctly rounded 9-decimal value.
fn round9(v: f64) -> f64 {
    format!("{v:.9}").parse().unwrap()
}
