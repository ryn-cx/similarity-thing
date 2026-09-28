package main

import (
	"math"
	"testing"
)

func r(s string) []rune { return []rune(s) }

func near(x, y float64) bool { return math.Abs(x-y) < 1e-3 }

func TestGrams(t *testing.T) {
	// night: ni ig gh ht; nacht: na ac ch ht -> 1 shared of 7 distinct
	if got := jaccardSimilarity(r("night"), r("nacht"), 2); !near(got, 1.0/7) {
		t.Errorf("jaccard = %f", got)
	}
	if got := sorensenDice(r("night"), r("nacht"), 2); !near(got, 0.25) {
		t.Errorf("dice = %f", got)
	}
	if got := cosineSimilarity(r("night"), r("nacht"), 2); !near(got, 0.25) {
		t.Errorf("cosine = %f", got)
	}
	if d, total := qgramDistance(r("abcd"), r("abce"), 3); d != 2 || total != 4 {
		t.Errorf("qgram = %d/%d", d, total)
	}
	if got := jaccardSimilarity(r(""), r(""), 2); got != 1 {
		t.Errorf("empty jaccard = %f", got)
	}
}

func TestSequenceAlignment(t *testing.T) {
	if n, start := longestCommonSubstring(r("The quick brown fox"), r("A fast brown fox")); n != 10 || start != 9 {
		t.Errorf("longest common substring = %d at %d", n, start)
	}
	// expected values from Python difflib.SequenceMatcher(autojunk=False)
	for _, c := range []struct {
		a, b string
		want float64
	}{
		{"The quick brown fox", "A fast brown fox", 0.6285714285714286},
		{"kitten", "sitting", 0.6153846153846154},
		{"GESTALT PATTERN MATCHING", "GESTALT PRACTICE", 0.6},
		{"", "", 1},
	} {
		if got := ratcliffObershelp(r(c.a), r(c.b)); !near(got, c.want) {
			t.Errorf("ratcliff(%q,%q) = %f want %f", c.a, c.b, got, c.want)
		}
	}
	// GATTACA / GCATGCU: textbook global alignment score 0 with +1/-1/-1
	if got := needlemanWunsch(r("GATTACA"), r("GCATGCU")); got != 0 {
		t.Errorf("needleman-wunsch = %d", got)
	}
	if got := needlemanWunsch(r("abc"), r("abc")); got != 3 {
		t.Errorf("needleman-wunsch identical = %d", got)
	}
	if got := overlapCoefficient(r("brown"), r("the brown fox"), 2); got != 1 {
		t.Errorf("overlap contained = %f", got)
	}
}

func TestTFIDF(t *testing.T) {
	// shared "fox" has idf 1, the unshared words ln(1.5)+1 each
	idf := math.Log(1.5) + 1
	want := 1 / (math.Sqrt(1+idf*idf) * math.Sqrt(1+idf*idf))
	if got := tfidfCosine(r("red fox"), r("fox blue")); !near(got, want) {
		t.Errorf("tfidf = %f want %f", got, want)
	}
	if got := tfidfCosine(r("the cat sat"), r("sat, the cat!")); !near(got, 1) {
		t.Errorf("tfidf reordered = %f", got)
	}
	if got := tfidfCosine(r("alpha"), r("beta")); got != 0 {
		t.Errorf("tfidf disjoint = %f", got)
	}
}

func TestPhonetic(t *testing.T) {
	cologneCases := map[string]string{
		"MUELLER": "657", "MULLER": "657", "MEYER": "67", "MAIER": "67",
		"WIKIPEDIA": "3412", "BREMERHAVEN": "176736", "CHRISTIAN": "47826",
	}
	for in, want := range cologneCases {
		if got := colognePhonetic(in); got != want {
			t.Errorf("cologne(%s)=%s want %s", in, got, want)
		}
	}
	metaphoneCases := map[string]string{
		"THOMPSON": "0MPSN", "KNIGHT": "NT", "PHONE": "FN", "SCHOOL": "SKL",
		"WHITE": "WT", "XRAY": "SR", "DODGE": "TJ", "SCIENCE": "SNS",
	}
	for in, want := range metaphoneCases {
		if got := metaphone(in); got != want {
			t.Errorf("metaphone(%s)=%s want %s", in, got, want)
		}
	}
	if got := phoneticEncode("Müller-Lüdenscheidt", colognePhonetic); got != "657 52682" {
		t.Errorf("multi-word cologne = %q", got)
	}
}

func TestCompareStrings(t *testing.T) {
	res := compareStrings("Kitten", "sitting", true)
	if len(res) != 24 {
		t.Fatalf("got %d results", len(res))
	}
	for _, x := range res {
		if x.Name == "Hamming Distance" {
			if x.Value != "n/a" || x.Score != nil {
				t.Errorf("hamming on unequal lengths = %+v", x)
			}
			continue
		}
		if x.Status != "ok" || x.Score == nil || *x.Score < 0 || *x.Score > 1 || x.Nanos <= 0 {
			t.Errorf("bad result %+v", x)
		}
	}
}
