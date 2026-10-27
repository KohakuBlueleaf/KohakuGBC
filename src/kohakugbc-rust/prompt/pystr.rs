//! Python's own whitespace and line rules (`str.isspace`, `strip`, `splitlines`), so text reads
//! here as `kohakugbc.graph.prompt` reads it.

/// `str.isspace`: Rust's whitespace plus the separators U+001C..U+001F.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `str.splitlines()`: `\r\n` is one break.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let brk = matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if !brk {
            continue;
        }
        out.push(&s[start..i]);
        let mut next = i + c.len_utf8();
        if c == '\r' {
            if let Some(&(j, '\n')) = it.peek() {
                it.next();
                next = j + 1;
            }
        }
        start = next;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `re.sub(r"\s+", " ", s).strip()`.
pub fn collapse(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if is_space(c) {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(c);
    }
    out
}
