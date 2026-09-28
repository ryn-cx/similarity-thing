package main

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/antzucaro/matchr"
)

// Result is one row of the comparison table.
type Result struct {
	Group  string   `json:"group"`
	Name   string   `json:"name"`
	Value  string   `json:"value"`
	Detail string   `json:"detail,omitempty"` // describes the method; shown under the name
	Note   string   `json:"note,omitempty"`   // extra result info; shown with the value
	Score  *float64 `json:"score"`            // 0..1 similarity, nil when not applicable
	Match  *bool    `json:"match"`            // phonetic code equality
	Nanos  int64    `json:"nanos"`            // mean wall time of one computation
	Runs   int      `json:"runs"`             // how many runs the mean is taken over
	Status string   `json:"status"`           // ok | loading | error
}

const (
	groupEdit     = "Edit distance"
	groupSequence = "Sequence alignment"
	groupGram     = "Token & q-gram"
	groupPhonetic = "Phonetic"
	groupSemantic = "Semantic embeddings"
)

// clockResolution is the smallest step the monotonic clock advances by:
// nanoseconds on Linux and macOS, but around half a millisecond on Windows.
var clockResolution = func() time.Duration {
	best := time.Hour
	for range 5 {
		start := time.Now()
		for time.Since(start) == 0 {
		}
		best = min(best, time.Since(start))
	}
	return best
}()

// measureBudget is how long measure keeps re-running a fast algorithm. It
// must span many clock ticks for the mean to be meaningful.
var measureBudget = max(time.Millisecond, 20*clockResolution)

// measure runs fn until it has either taken a meaningful amount of wall time
// or hit a run cap, and returns the mean duration per run. Very fast
// algorithms finish in tens of nanoseconds, well below timer resolution, so
// a single sample would be mostly noise.
func measure(fn func()) (time.Duration, int) {
	const maxRuns = 200_000
	start := time.Now()
	fn()
	first := time.Since(start)
	if first >= measureBudget/4 {
		return first, 1
	}
	runs := 1
	for time.Since(start) < measureBudget && runs < maxRuns {
		fn()
		runs++
	}
	return time.Since(start) / time.Duration(runs), runs
}

func f64(v float64) *float64 { return &v }

func fmtScore(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) }

// compareStrings runs every classic string algorithm over a and b.
func compareStrings(a, b string, ignoreCase bool) []Result {
	ra, rb := prepare(a, ignoreCase), prepare(b, ignoreCase)
	sa, sb := string(ra), string(rb)
	maxLen := max(len(ra), len(rb))
	var results []Result

	distance := func(group, name string, fn func(x, y string) int, norm int) {
		var d int
		dur, runs := measure(func() { d = fn(sa, sb) })
		sim := normalizedDistance(d, norm)
		results = append(results, Result{
			Group: group, Name: name,
			Value: strconv.Itoa(d),
			Note:  fmt.Sprintf("%s similarity", fmtScore(sim)),
			Score: f64(sim), Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok",
		})
	}
	similarity := func(group, name, detail string, fn func() float64) {
		var s float64
		dur, runs := measure(func() { s = fn() })
		results = append(results, Result{
			Group: group, Name: name,
			Value: fmtScore(s), Detail: detail,
			Score: f64(s), Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok",
		})
	}
	phonetic := func(name string, enc func(string) string) {
		var ca, cb string
		dur, runs := measure(func() {
			ca, cb = phoneticEncode(a, enc), phoneticEncode(b, enc)
		})
		match := ca == cb
		score := 0.0
		if match {
			score = 1
		}
		results = append(results, Result{
			Group: groupPhonetic, Name: name,
			Value: orDash(ca) + "  ·  " + orDash(cb),
			Score: f64(score), Match: &match,
			Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok",
		})
	}

	distance(groupEdit, "Levenshtein Distance", matchr.Levenshtein, maxLen)
	distance(groupEdit, "OSA Damerau-Levenshtein", matchr.OSA, maxLen)
	distance(groupEdit, "Damerau-Levenshtein", matchr.DamerauLevenshtein, maxLen)
	// only insertions and deletions: every character outside the longest
	// common subsequence is removed from one side or inserted into the other
	distance(groupEdit, "LCS Edit Distance", func(x, y string) int {
		return len(ra) + len(rb) - 2*matchr.LongestCommonSubsequence(x, y)
	}, len(ra)+len(rb))
	{
		var d int
		var err error
		dur, runs := measure(func() { d, err = matchr.Hamming(sa, sb) })
		r := Result{Group: groupEdit, Name: "Hamming Distance", Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok"}
		if err == nil {
			sim := normalizedDistance(d, maxLen)
			r.Value, r.Note, r.Score = strconv.Itoa(d), fmt.Sprintf("%s similarity", fmtScore(sim)), f64(sim)
		} else {
			r.Value, r.Note = "n/a", fmt.Sprintf("needs equal lengths (%d vs %d)", len(ra), len(rb))
			r.Nanos, r.Runs = 0, 0 // nothing was computed
		}
		results = append(results, r)
	}

	similarity(groupSequence, "Jaro Similarity", "", func() float64 { return matchr.Jaro(sa, sb) })
	similarity(groupSequence, "Jaro-Winkler", "prefix scale 0.1", func() float64 { return matchr.JaroWinkler(sa, sb, false) })
	similarity(groupSequence, "Jaro-Winkler (long tolerance)", "prefix scale 0.1 · extra boost for long strings", func() float64 { return matchr.JaroWinkler(sa, sb, true) })
	similarity(groupSequence, "Ratcliff/Obershelp", "Gestalt pattern matching · difflib ratio", func() float64 { return ratcliffObershelp(ra, rb) })
	{
		var n, start int
		dur, runs := measure(func() { n, start = longestCommonSubstring(ra, rb) })
		sim := normalizedDistance(maxLen-n, maxLen)
		note := "nothing in common"
		if n > 0 {
			note = "“" + truncate(string(ra[start:start+n]), 40) + "”"
		}
		note += " · " + fmtScore(sim) + " similarity"
		results = append(results, Result{
			Group: groupSequence, Name: "Longest Common Substring",
			Value: strconv.Itoa(n), Note: note,
			Score: f64(sim), Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok",
		})
	}
	alignment := func(name, detail string, fn func() float64, best int) {
		var score float64
		dur, runs := measure(func() { score = fn() })
		sim := 1.0
		if best > 0 {
			sim = min(1, max(0, score)/float64(best))
		} else if maxLen > 0 {
			sim = 0
		}
		results = append(results, Result{
			Group: groupSequence, Name: name,
			Value:  strconv.FormatFloat(score, 'f', -1, 64),
			Detail: detail,
			Note:   fmt.Sprintf("%s similarity", fmtScore(sim)),
			Score:  f64(sim), Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok",
		})
	}
	alignment("Needleman-Wunsch", "global score · match +1, mismatch −1, gap −1",
		func() float64 { return float64(needlemanWunsch(ra, rb)) }, maxLen)
	alignment("Smith-Waterman", "local score · match +1, mismatch −2, gap −0.5",
		func() float64 {
			// matchr returns the other string's length when one side is empty
			if len(ra) == 0 || len(rb) == 0 {
				return 0
			}
			return matchr.SmithWaterman(sa, sb)
		}, min(len(ra), len(rb)))

	similarity(groupGram, "Cosine Similarity", "character bigrams", func() float64 { return cosineSimilarity(ra, rb, 2) })
	similarity(groupGram, "Jaccard Similarity", "character bigrams", func() float64 { return jaccardSimilarity(ra, rb, 2) })
	similarity(groupGram, "Sørensen-Dice", "character bigrams", func() float64 { return sorensenDice(ra, rb, 2) })
	similarity(groupGram, "Overlap Coefficient", "character bigrams", func() float64 { return overlapCoefficient(ra, rb, 2) })
	{
		var dist, total int
		dur, runs := measure(func() { dist, total = qgramDistance(ra, rb, 3) })
		sim := 1.0
		if total > 0 {
			sim = 1 - float64(dist)/float64(total)
		}
		results = append(results, Result{
			Group: groupGram, Name: "Q-gram Similarity",
			Value:  fmtScore(sim),
			Detail: "q = 3",
			Note:   fmt.Sprintf("distance %d", dist),
			Score:  f64(sim), Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok",
		})
	}
	similarity(groupGram, "TF-IDF Cosine", "word tokens · smoothed IDF", func() float64 { return tfidfCosine(ra, rb) })

	phonetic("Soundex", matchr.Soundex)
	phonetic("Phonex", matchr.Phonex)
	phonetic("Cologne Phonetic", colognePhonetic)
	phonetic("Metaphone", metaphone)
	{
		var pa, aa, pb, ab string
		dur, runs := measure(func() {
			pa, aa = doubleMetaphoneText(a)
			pb, ab = doubleMetaphoneText(b)
		})
		match := pa == pb || pa == ab || aa == pb || aa == ab
		score := 0.0
		if match {
			score = 1
		}
		r := Result{
			Group: groupPhonetic, Name: "Double Metaphone",
			Value: orDash(pa) + "  ·  " + orDash(pb),
			Score: f64(score), Match: &match,
			Nanos: dur.Nanoseconds(), Runs: runs, Status: "ok",
		}
		if aa != pa || ab != pb {
			r.Note = "alternates " + orDash(aa) + " · " + orDash(ab)
		}
		results = append(results, r)
	}
	phonetic("NYSIIS", matchr.NYSIIS)

	return results
}

// doubleMetaphoneText encodes every word of s, returning the primary and
// alternate codes.
func doubleMetaphoneText(s string) (string, string) {
	var p, a []string
	for _, w := range phoneticWords(s) {
		pw, aw := matchr.DoubleMetaphone(w)
		if pw != "" || aw != "" {
			p, a = append(p, orDash(pw)), append(a, orDash(aw))
		}
	}
	return strings.Join(p, " "), strings.Join(a, " ")
}

func truncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n-1]) + "…"
	}
	return s
}

func orDash(s string) string {
	if s == "" {
		return "—"
	}
	return s
}
