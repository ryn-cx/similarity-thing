package main

// Static embedding models in the sentence-transformers StaticEmbedding
// format: a token embedding table and a BERT WordPiece tokenizer. A text's
// embedding is the mean of its tokens' rows, like go-potion's models.

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// staticEmbedding is a loaded model. It is safe for concurrent use.
type staticEmbedding struct {
	vocab map[string]int
	unk   int
	table []float32 // [vocab][dims]
	dims  int
}

func (m *staticEmbedding) Dimensions() int { return m.dims }

// loadStaticEmbedding downloads a sentence-transformers StaticEmbedding
// model into the go-potion cache directory on first use and loads it.
func loadStaticEmbedding(ctx context.Context, repo string) (*staticEmbedding, error) {
	dir := os.Getenv("GO_POTION_HOME")
	if dir == "" {
		c, err := os.UserCacheDir()
		if err != nil {
			return nil, err
		}
		dir = filepath.Join(c, "go-potion")
	}
	dir = filepath.Join(dir, path.Base(repo))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	for _, f := range []string{"model.safetensors", "tokenizer.json"} {
		url := "https://huggingface.co/" + repo + "/resolve/main/0_StaticEmbedding/" + f
		if err := fetchOnce(ctx, url, filepath.Join(dir, f)); err != nil {
			return nil, fmt.Errorf("download %s: %w", f, err)
		}
	}

	m := &staticEmbedding{}
	b, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	var tok struct {
		Model struct {
			Type  string         `json:"type"`
			Unk   string         `json:"unk_token"`
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(b, &tok); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if tok.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("tokenizer.json: model type %q, want WordPiece", tok.Model.Type)
	}
	m.vocab = tok.Model.Vocab
	var ok bool
	if m.unk, ok = m.vocab[tok.Model.Unk]; !ok {
		return nil, fmt.Errorf("tokenizer.json: no unknown token")
	}

	var rows int
	if m.table, rows, m.dims, err = readEmbeddingTable(filepath.Join(dir, "model.safetensors"), "embedding.weight"); err != nil {
		return nil, err
	}
	for _, id := range m.vocab {
		if id < 0 || id >= rows {
			return nil, fmt.Errorf("vocabulary id %d outside the %d-row embedding table", id, rows)
		}
	}
	return m, nil
}

// Encode returns the L2-normalised mean of the text's token embeddings, or a
// zero vector if it has no tokens.
func (m *staticEmbedding) Encode(_ context.Context, s string) ([]float32, error) {
	out := make([]float32, m.dims)
	ids := m.tokenize(s)
	for _, id := range ids {
		for i, v := range m.table[id*m.dims : (id+1)*m.dims] {
			out[i] += v
		}
	}
	var sum float64
	for _, v := range out {
		sum += float64(v) * float64(v)
	}
	if norm := math.Sqrt(sum); norm > 0 {
		for i := range out {
			out[i] = float32(float64(out[i]) / norm)
		}
	}
	return out, nil
}

// tokenize applies BERT's uncased basic tokenizer and then WordPiece,
// returning vocabulary ids without [CLS]/[SEP], as StaticEmbedding does.
func (m *staticEmbedding) tokenize(s string) []int {
	var b strings.Builder
	for _, r := range norm.NFD.String(strings.ToLower(s)) {
		switch {
		case r == 0 || r == unicode.ReplacementChar || isBertControl(r):
		case unicode.Is(unicode.Mn, r): // strip accents
		case isBertSpace(r):
			b.WriteByte(' ')
		case isBertPunct(r) || isCJK(r):
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	var ids []int
	for _, word := range strings.Fields(b.String()) {
		ids = append(ids, m.wordPiece(word)...)
	}
	return ids
}

func (m *staticEmbedding) wordPiece(word string) []int {
	runes := []rune(word)
	if len(runes) > 100 {
		return []int{m.unk}
	}
	var ids []int
	for start := 0; start < len(runes); {
		end, id := len(runes), -1
		for ; end > start; end-- {
			piece := string(runes[start:end])
			if start > 0 {
				piece = "##" + piece
			}
			if v, ok := m.vocab[piece]; ok {
				id = v
				break
			}
		}
		if id < 0 {
			return []int{m.unk}
		}
		ids = append(ids, id)
		start = end
	}
	return ids
}

func isBertSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || unicode.Is(unicode.Zs, r)
}

func isBertControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.In(r, unicode.Cc, unicode.Cf, unicode.Co, unicode.Cs)
}

func isBertPunct(r rune) bool {
	return (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) ||
		unicode.IsPunct(r)
}

func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) || (r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) || (r >= 0x2B740 && r <= 0x2B81F) || (r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) || (r >= 0x2F800 && r <= 0x2FA1F)
}

// readEmbeddingTable reads one 2-D F32 tensor from a safetensors file.
func readEmbeddingTable(file, name string) (table []float32, rows, dims int, err error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, 0, 0, err
	}
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil || n > uint64(st.Size()-8) {
		return nil, 0, 0, fmt.Errorf("%s: bad header", file)
	}
	header := make([]byte, n)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", file, err)
	}
	var tensors map[string]json.RawMessage
	if err := json.Unmarshal(header, &tensors); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", file, err)
	}
	var info struct {
		Dtype   string   `json:"dtype"`
		Shape   []int    `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}
	if raw, ok := tensors[name]; !ok {
		return nil, 0, 0, fmt.Errorf("%s: missing tensor %s", file, name)
	} else if err := json.Unmarshal(raw, &info); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", file, err)
	}
	if info.Dtype != "F32" || len(info.Shape) != 2 || info.Offsets[0] < 0 ||
		info.Offsets[1]-info.Offsets[0] != int64(info.Shape[0])*int64(info.Shape[1])*4 ||
		8+int64(n)+info.Offsets[1] > st.Size() {
		return nil, 0, 0, fmt.Errorf("%s: unsupported tensor %s", file, name)
	}
	b := make([]byte, info.Offsets[1]-info.Offsets[0])
	if _, err := f.ReadAt(b, 8+int64(n)+info.Offsets[0]); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", file, err)
	}
	table = make([]float32, len(b)/4)
	for i := range table {
		table[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return table, info.Shape[0], info.Shape[1], nil
}

// fetchOnce downloads url to dest unless dest exists, via a temp file so an
// interrupted download never leaves a partial file behind.
func fetchOnce(ctx context.Context, url, dest string) error {
	if _, err := os.Stat(dest); err == nil {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dest), filepath.Base(dest)+".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), dest)
}
