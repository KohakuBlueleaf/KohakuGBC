//! Box coordinates as written in a prompt, and the header that declares them
//! (`kohakugbc.graph.prompt.coords`).

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
}

impl Default for Coords {
    fn default() -> Self {
        Coords {
            width: 1024,
            height: 1024,
            space: "abs_centralized".into(),
            order: "yxyx".into(),
            scale: 1000,
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
    ) -> Result<Self, String> {
        let c = Coords {
            width,
            height,
            space: space.into(),
            order: order.into(),
            scale,
        };
        if !matches!(space, "abs_centralized" | "relative") || !matches!(order, "yxyx" | "xyxy") {
            return Err(format!(
                "space in (abs_centralized, relative), order in (yxyx, xyxy): {c:?}"
            ));
        }
        if width.min(height).min(scale) <= 0 {
            return Err(format!("width, height and scale are positive: {c:?}"));
        }
        Ok(c)
    }

    /// (size, offset) of x, then of y: written = round(stored * size - offset).
    fn axes(&self) -> ((f64, f64), (f64, f64)) {
        if self.space == "relative" {
            let s = self.scale as f64;
            return ((s, 0.0), (s, 0.0));
        }
        let (w, h) = (self.width as f64, self.height as f64);
        ((w, w / 2.0), (h, h / 2.0))
    }

    /// Python's `round`: half to even.
    pub fn write(&self, b: &Box4) -> [i64; 4] {
        let ((sx, ox), (sy, oy)) = self.axes();
        let r = |x: f64| x.round_ties_even() as i64;
        let (x0, y0, x1, y1) = (
            r(b[0] * sx - ox),
            r(b[1] * sy - oy),
            r(b[2] * sx - ox),
            r(b[3] * sy - oy),
        );
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

        // An odd side's edge rounds half a pixel out (+-771.5 -> +-772): clamp that back to
        // [0, 1]; anything further out is off the canvas
        let at = |v: i64, s: f64, o: f64| -> Result<f64, String> {
            let p = v as f64 + o;
            if !(-0.5 <= p && p <= s + 0.5) {
                return Err(format!("{ints:?}: off the canvas of {}", self.header()));
            }
            Ok(((p) / s).clamp(0.0, 1.0))
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
        )
    }
}
