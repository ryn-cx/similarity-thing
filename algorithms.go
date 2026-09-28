package main

import (
	"math"
	"strings"
	"unicode"
	"unicode/utf8"
)

// longestCommonSubstring returns the length of the longest run of characters
// shared by a and b, and where it starts in a.
func longestCommonSubstring(a, b []rune) (length, start int) {
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			if a[i-1] == b[j-1] {
				curr[j] = prev[j-1] + 1
				if curr[j] > length {
					length, start = curr[j], i-curr[j]
				}
			} else {
				curr[j] = 0
			}
		}
		prev, curr = curr, prev
	}
	return length, start
}

// ratcliffObershelp is the Gestalt pattern matching ratio 2·M / (|a|+|b|),
// where M counts the characters in the longest common substring plus,
// recursively, the matches to its left and right. It reproduces Python's
// difflib.SequenceMatcher(None, a, b, autojunk=False).ratio().
func ratcliffObershelp(a, b []rune) float64 {
	if len(a)+len(b) == 0 {
		return 1
	}
	type span struct{ alo, ahi, blo, bhi int }
	matches := 0
	stack := []span{{0, len(a), 0, len(b)}}
	row := make([]int, len(b)+1)
	for len(stack) > 0 {
		s := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		// longest match in a[alo:ahi] × b[blo:bhi]; earliest in a, then b, on ties
		besti, bestj, best := s.alo, s.blo, 0
		clear(row)
		for i := s.alo; i < s.ahi; i++ {
			diag := 0
			for j := s.blo; j < s.bhi; j++ {
				up := row[j+1]
				if a[i] == b[j] {
					row[j+1] = diag + 1
					if row[j+1] > best {
						best, besti, bestj = row[j+1], i-row[j+1]+1, j-row[j+1]+1
					}
				} else {
					row[j+1] = 0
				}
				diag = up
			}
		}
		if best == 0 {
			continue
		}
		matches += best
		if s.alo < besti && s.blo < bestj {
			stack = append(stack, span{s.alo, besti, s.blo, bestj})
		}
		if besti+best < s.ahi && bestj+best < s.bhi {
			stack = append(stack, span{besti + best, s.ahi, bestj + best, s.bhi})
		}
	}
	return 2 * float64(matches) / float64(len(a)+len(b))
}

// Needleman-Wunsch alignment scores.
const (
	alignMatch    = 1
	alignMismatch = -1
	alignGap      = -1
)

// needlemanWunsch returns the best global alignment score of a and b.
func needlemanWunsch(a, b []rune) int {
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j * alignGap
	}
	for i := 1; i <= len(a); i++ {
		curr[0] = i * alignGap
		for j := 1; j <= len(b); j++ {
			sub := alignMismatch
			if a[i-1] == b[j-1] {
				sub = alignMatch
			}
			curr[j] = max(prev[j-1]+sub, prev[j]+alignGap, curr[j-1]+alignGap)
		}
		prev, curr = curr, prev
	}
	return prev[len(b)]
}

// ---------------------------------------------------------------------------
// Q-gram / token based measures
// ---------------------------------------------------------------------------

// shingles counts the character q-grams of s. Strings shorter than q (but
// non-empty) become a single shingle so they still compare meaningfully.
func shingles(s []rune, q int) map[string]int {
	out := make(map[string]int)
	if len(s) == 0 {
		return out
	}
	if len(s) < q {
		out[string(s)]++
		return out
	}
	for i := 0; i+q <= len(s); i++ {
		out[string(s[i:i+q])]++
	}
	return out
}

func cosineSimilarity(a, b []rune, q int) float64 {
	sa, sb := shingles(a, q), shingles(b, q)
	if len(sa) == 0 && len(sb) == 0 {
		return 1
	}
	var dot, na, nb float64
	for k, va := range sa {
		na += float64(va * va)
		dot += float64(va * sb[k])
	}
	for _, vb := range sb {
		nb += float64(vb * vb)
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func setOverlap(a, b []rune, q int) (inter, sizeA, sizeB int) {
	sa, sb := shingles(a, q), shingles(b, q)
	for k := range sa {
		if _, ok := sb[k]; ok {
			inter++
		}
	}
	return inter, len(sa), len(sb)
}

// overlapCoefficient (Szymkiewicz-Simpson) divides the shared q-grams by
// the size of the smaller set, so a string contained in another scores 1.
func overlapCoefficient(a, b []rune, q int) float64 {
	inter, na, nb := setOverlap(a, b, q)
	if na == 0 && nb == 0 {
		return 1
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float64(inter) / float64(min(na, nb))
}

func jaccardSimilarity(a, b []rune, q int) float64 {
	inter, na, nb := setOverlap(a, b, q)
	union := na + nb - inter
	if union == 0 {
		return 1
	}
	return float64(inter) / float64(union)
}

func sorensenDice(a, b []rune, q int) float64 {
	inter, na, nb := setOverlap(a, b, q)
	if na+nb == 0 {
		return 1
	}
	return 2 * float64(inter) / float64(na+nb)
}

// qgramDistance is Ukkonen's q-gram distance: the L1 distance between the
// q-gram profiles of both strings.
func qgramDistance(a, b []rune, q int) (dist, total int) {
	sa, sb := shingles(a, q), shingles(b, q)
	for k, va := range sa {
		dist += abs(va - sb[k])
		total += va
	}
	for k, vb := range sb {
		if _, ok := sa[k]; !ok {
			dist += vb
		}
		total += vb
	}
	return dist, total
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// words splits s into runs of letters and digits.
func words(s []rune) []string {
	var out []string
	start := -1
	for i, r := range s {
		isWord := unicode.IsLetter(r) || unicode.IsDigit(r)
		if isWord && start < 0 {
			start = i
		} else if !isWord && start >= 0 {
			out = append(out, string(s[start:i]))
			start = -1
		}
	}
	if start >= 0 {
		out = append(out, string(s[start:]))
	}
	return out
}

// tfidfCosine treats a and b as a two-document corpus, weights each word by
// term frequency × inverse document frequency and returns the cosine of the
// two vectors. IDF is smoothed as in scikit-learn, ln((1+N)/(1+df)) + 1, so
// words shared by both texts still count (with plain IDF they would weigh 0
// in a two-document corpus).
func tfidfCosine(a, b []rune) float64 {
	tfA, tfB := map[string]int{}, map[string]int{}
	for _, w := range words(a) {
		tfA[w]++
	}
	for _, w := range words(b) {
		tfB[w]++
	}
	if len(tfA) == 0 && len(tfB) == 0 {
		return 1
	}
	if len(tfA) == 0 || len(tfB) == 0 {
		return 0
	}
	idfShared, idfSingle := math.Log(3.0/3.0)+1, math.Log(3.0/2.0)+1
	var dot, na, nb float64
	for w, ca := range tfA {
		idf := idfSingle
		if cb, ok := tfB[w]; ok {
			idf = idfShared
			dot += float64(ca) * idf * float64(cb) * idf
		}
		na += math.Pow(float64(ca)*idf, 2)
	}
	for w, cb := range tfB {
		idf := idfSingle
		if _, ok := tfA[w]; ok {
			idf = idfShared
		}
		nb += math.Pow(float64(cb)*idf, 2)
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func normalizedDistance(dist, maxLen int) float64 {
	if maxLen == 0 {
		return 1
	}
	return 1 - float64(dist)/float64(maxLen)
}

func prepare(s string, ignoreCase bool) []rune {
	if ignoreCase {
		s = strings.ToLower(s)
	}
	out := make([]rune, 0, utf8.RuneCountInString(s))
	for _, r := range s {
		out = append(out, r)
	}
	return out
}
