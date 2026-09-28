package main

import (
	"strings"
	"unicode"
)

// phoneticWords upper-cases s, folds common German/Latin accents to ASCII
// and splits it into runs of letters A-Z.
func phoneticWords(s string) []string {
	var words []string
	var b strings.Builder
	flush := func() {
		if b.Len() > 0 {
			words = append(words, b.String())
			b.Reset()
		}
	}
	for _, r := range strings.ToUpper(s) {
		switch r {
		case 'Ä', 'À', 'Á', 'Â', 'Ã', 'Å':
			b.WriteByte('A')
		case 'Ö', 'Ò', 'Ó', 'Ô', 'Õ', 'Ø':
			b.WriteByte('O')
		case 'Ü', 'Ù', 'Ú', 'Û':
			b.WriteByte('U')
		case 'È', 'É', 'Ê', 'Ë':
			b.WriteByte('E')
		case 'Ì', 'Í', 'Î', 'Ï':
			b.WriteByte('I')
		case 'Ç':
			b.WriteByte('C')
		case 'Ñ':
			b.WriteByte('N')
		case 'ß', 'ẞ':
			b.WriteString("SS")
		default:
			if r >= 'A' && r <= 'Z' {
				b.WriteRune(r)
			} else if !unicode.IsLetter(r) {
				flush()
			}
		}
	}
	flush()
	return words
}

// phoneticEncode applies a single-word encoder to every word in s.
func phoneticEncode(s string, enc func(string) string) string {
	var codes []string
	for _, w := range phoneticWords(s) {
		if c := enc(w); c != "" {
			codes = append(codes, c)
		}
	}
	return strings.Join(codes, " ")
}

// ---------------------------------------------------------------------------
// Cologne phonetics (Kölner Phonetik)
// ---------------------------------------------------------------------------

func colognePhonetic(w string) string {
	at := func(i int) byte {
		if i < 0 || i >= len(w) {
			return 0
		}
		return w[i]
	}
	in := func(c byte, set string) bool { return c != 0 && strings.IndexByte(set, c) >= 0 }

	var raw []byte
	for i := 0; i < len(w); i++ {
		c, prev, next := w[i], at(i-1), at(i+1)
		switch c {
		case 'A', 'E', 'I', 'J', 'O', 'U', 'Y':
			raw = append(raw, '0')
		case 'H':
			// no code
		case 'B':
			raw = append(raw, '1')
		case 'P':
			if next == 'H' {
				raw = append(raw, '3')
			} else {
				raw = append(raw, '1')
			}
		case 'D', 'T':
			if in(next, "CSZ") {
				raw = append(raw, '8')
			} else {
				raw = append(raw, '2')
			}
		case 'F', 'V', 'W':
			raw = append(raw, '3')
		case 'G', 'K', 'Q':
			raw = append(raw, '4')
		case 'C':
			if i == 0 {
				if in(next, "AHKLOQRUX") {
					raw = append(raw, '4')
				} else {
					raw = append(raw, '8')
				}
			} else if in(next, "AHKOQUX") && !in(prev, "SZ") {
				raw = append(raw, '4')
			} else {
				raw = append(raw, '8')
			}
		case 'X':
			if in(prev, "CKQ") {
				raw = append(raw, '8')
			} else {
				raw = append(raw, '4', '8')
			}
		case 'L':
			raw = append(raw, '5')
		case 'M', 'N':
			raw = append(raw, '6')
		case 'R':
			raw = append(raw, '7')
		case 'S', 'Z':
			raw = append(raw, '8')
		}
	}

	// collapse repeated codes, then drop every 0 except a leading one
	var out []byte
	for i, c := range raw {
		if i > 0 && c == raw[i-1] {
			continue
		}
		if c == '0' && i > 0 {
			continue
		}
		out = append(out, c)
	}
	return string(out)
}

// ---------------------------------------------------------------------------
// Metaphone (Lawrence Philips, 1990)
// ---------------------------------------------------------------------------

func isVowel(c byte) bool { return c != 0 && strings.IndexByte("AEIOU", c) >= 0 }

func metaphone(w string) string {
	if w == "" {
		return ""
	}
	switch {
	case len(w) > 1 && (strings.HasPrefix(w, "AE") || strings.HasPrefix(w, "GN") ||
		strings.HasPrefix(w, "KN") || strings.HasPrefix(w, "PN") || strings.HasPrefix(w, "WR")):
		w = w[1:]
	case w[0] == 'X':
		w = "S" + w[1:]
	case strings.HasPrefix(w, "WH"):
		w = "W" + w[2:]
	}

	at := func(i int) byte {
		if i < 0 || i >= len(w) {
			return 0
		}
		return w[i]
	}
	in := func(c byte, set string) bool { return c != 0 && strings.IndexByte(set, c) >= 0 }

	var out []byte
	for i := 0; i < len(w); i++ {
		c, prev, next, next2 := w[i], at(i-1), at(i+1), at(i+2)
		if c == prev && c != 'C' {
			continue
		}
		switch c {
		case 'A', 'E', 'I', 'O', 'U':
			if i == 0 {
				out = append(out, c)
			}
		case 'B':
			if !(i == len(w)-1 && prev == 'M') {
				out = append(out, 'B')
			}
		case 'C':
			switch {
			case prev == 'S' && in(next, "EIY"):
				// silent in -SCI-, -SCE-, -SCY-
			case next == 'I' && next2 == 'A':
				out = append(out, 'X')
			case next == 'H':
				if prev == 'S' {
					out = append(out, 'K')
				} else {
					out = append(out, 'X')
				}
			case in(next, "EIY"):
				out = append(out, 'S')
			default:
				out = append(out, 'K')
			}
		case 'D':
			if next == 'G' && in(next2, "EIY") {
				out = append(out, 'J')
			} else {
				out = append(out, 'T')
			}
		case 'G':
			switch {
			case next == 'H' && !isVowel(next2):
				// silent in -GH- unless followed by a vowel
			case prev == 'D' && in(next, "EIY"):
				// already voiced by the D in -DGE-, -DGI-, -DGY-
			case next == 'N' && (i+2 == len(w) || w[i+2:] == "ED"):
				// silent in -GN and -GNED
			case in(next, "EIY") && prev != 'G':
				out = append(out, 'J')
			default:
				out = append(out, 'K')
			}
		case 'H':
			if in(prev, "CSPTG") || (isVowel(prev) && !isVowel(next)) {
				continue
			}
			out = append(out, 'H')
		case 'K':
			if prev != 'C' {
				out = append(out, 'K')
			}
		case 'P':
			if next == 'H' {
				out = append(out, 'F')
			} else {
				out = append(out, 'P')
			}
		case 'Q':
			out = append(out, 'K')
		case 'S':
			if next == 'H' || (next == 'I' && in(next2, "OA")) {
				out = append(out, 'X')
			} else {
				out = append(out, 'S')
			}
		case 'T':
			switch {
			case next == 'I' && in(next2, "OA"):
				out = append(out, 'X')
			case next == 'H':
				out = append(out, '0')
			case next == 'C' && next2 == 'H':
				// silent in -TCH-
			default:
				out = append(out, 'T')
			}
		case 'V':
			out = append(out, 'F')
		case 'W', 'Y':
			if isVowel(next) {
				out = append(out, c)
			}
		case 'X':
			out = append(out, 'K', 'S')
		case 'Z':
			out = append(out, 'S')
		case 'F', 'J', 'L', 'M', 'N', 'R':
			out = append(out, c)
		}
	}
	return string(out)
}
