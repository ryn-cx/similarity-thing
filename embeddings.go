package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"path"
	"slices"
	"strings"
	"sync"
	"time"

	potion "github.com/trengrj/go-potion"
)

// encoder turns text into an embedding vector.
type encoder interface {
	Encode(context.Context, string) ([]float32, error)
	Dimensions() int
}

type potionEncoder struct{ *potion.Potion }

func (p potionEncoder) Encode(_ context.Context, s string) ([]float32, error) {
	return p.Potion.Encode(s)
}

// modelInfo describes one embedding model.
type modelInfo struct {
	ID     string
	Label  string
	SizeMB int
	Kind   string // static
	load   func(context.Context) (encoder, error)
}

func potionModel(id potion.Model, label string, sizeMB int) modelInfo {
	return modelInfo{string(id), label, sizeMB, "static", func(ctx context.Context) (encoder, error) {
		p, err := potion.New(ctx, id)
		if err != nil {
			return nil, err
		}
		return potionEncoder{p}, nil
	}}
}

func staticModel(id, repo string, sizeMB int) modelInfo {
	return modelInfo{id, path.Base(repo), sizeMB, "static", func(ctx context.Context) (encoder, error) {
		return loadStaticEmbedding(ctx, repo)
	}}
}

var allModels = []modelInfo{
	potionModel(potion.BASE2M, "potion-base-2M", 8),
	potionModel(potion.BASE4M, "potion-base-4M", 16),
	potionModel(potion.BASE8M, "potion-base-8M", 31),
	potionModel(potion.BASE32M, "potion-base-32M", 131),
	potionModel(potion.RETRIEVAL32M, "potion-retrieval-32M", 131),
	potionModel(potion.SCIENCE32M, "potion-science-32M", 130),
	potionModel(potion.CODE16M, "potion-code-16M", 65),
	potionModel(potion.CODE16MV2, "potion-code-16M-v2", 34),
	potionModel(potion.MULTILINGUAL128M, "potion-multilingual-128M", 531),
	staticModel("SIMILARITY-MULTILINGUAL", "sentence-transformers/static-similarity-mrl-multilingual-v1", 414),
}

// fmtSize formats a size in megabytes.
func fmtSize(mb int) string {
	if mb >= 1024 {
		return fmt.Sprintf("%.1f GB", float64(mb)/1024)
	}
	return fmt.Sprintf("%d MB", mb)
}

type modelSlot struct {
	info    modelInfo
	state   string // loading | ready | error
	err     string
	encoder encoder
}

// ModelStatus is the public view of a model slot.
type ModelStatus struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Size  string `json:"size"`
	Dims  int    `json:"dims,omitempty"`
	State string `json:"state"`
	Error string `json:"error,omitempty"`
}

// embedder owns the embedding models, which are downloaded and loaded in the
// background so the site is usable immediately.
type embedder struct {
	mu    sync.RWMutex
	slots []*modelSlot
}

func newEmbedder(selected string) (*embedder, error) {
	e := &embedder{}
	want := map[string]bool{}
	if s := strings.TrimSpace(selected); s != "" && !strings.EqualFold(s, "all") {
		for _, id := range strings.Split(s, ",") {
			want[strings.ToUpper(strings.TrimSpace(id))] = true
		}
	}
	all := len(want) == 0
	for _, m := range allModels {
		if all || want[m.ID] {
			e.slots = append(e.slots, &modelSlot{info: m, state: "loading"})
			delete(want, m.ID)
		}
	}
	for id := range want {
		if id != "" && id != "NONE" {
			return nil, fmt.Errorf("unknown model %q", id)
		}
	}
	return e, nil
}

// loadAll loads models one at a time, smallest first, so the quick ones
// become available while the big ones are still downloading.
func (e *embedder) loadAll(ctx context.Context) {
	order := slices.Clone(e.slots)
	slices.SortStableFunc(order, func(x, y *modelSlot) int { return x.info.SizeMB - y.info.SizeMB })
	for _, slot := range order {
		start := time.Now()
		enc, err := slot.info.load(ctx)
		e.mu.Lock()
		if err != nil {
			slot.state, slot.err = "error", err.Error()
			log.Printf("model %s failed: %v", slot.info.Label, err)
		} else {
			slot.state, slot.encoder = "ready", enc
			log.Printf("model %s ready (%d dims) in %s", slot.info.Label, enc.Dimensions(), time.Since(start).Round(time.Millisecond))
		}
		e.mu.Unlock()
	}
}

func (e *embedder) status() []ModelStatus {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]ModelStatus, 0, len(e.slots))
	for _, s := range e.slots {
		ms := ModelStatus{ID: s.info.ID, Label: s.info.Label, Size: fmtSize(s.info.SizeMB), State: s.state, Error: s.err}
		if s.encoder != nil {
			ms.Dims = s.encoder.Dimensions()
		}
		out = append(out, ms)
	}
	return out
}

// compare returns a row per configured model.
func (e *embedder) compare(a, b string) []Result {
	e.mu.RLock()
	slots := make([]modelSlot, len(e.slots))
	for i, s := range e.slots {
		slots[i] = *s
	}
	e.mu.RUnlock()

	results := make([]Result, 0, len(slots))
	for _, s := range slots {
		results = append(results, embedRow(context.Background(), s, a, b))
	}
	return results
}

// embedRow encodes a and b with one model. The timing covers encoding both
// texts plus the dot product.
func embedRow(ctx context.Context, s modelSlot, a, b string) Result {
	r := Result{Group: groupSemantic, Name: s.info.Label, Status: s.state}
	switch s.state {
	case "loading":
		r.Value = "Loading model…"
		r.Note = fmtSize(s.info.SizeMB) + " download on first start"
	case "error":
		r.Value = "Unavailable"
		r.Note = s.err
	case "ready":
		var cos float64
		var encErr error
		dur, runs := measure(func() {
			va, err := s.encoder.Encode(ctx, a)
			if err != nil {
				encErr = err
				return
			}
			vb, err := s.encoder.Encode(ctx, b)
			if err != nil {
				encErr = err
				return
			}
			cos = cosineOfEmbeddings(va, vb)
		})
		if encErr != nil {
			r.Status, r.Value, r.Note = "error", "Encode failed", encErr.Error()
			break
		}
		r.Status = "ok"
		r.Value = fmtScore(cos)
		r.Detail = fmt.Sprintf("%d-d cosine · %s", s.encoder.Dimensions(), s.info.Kind)
		r.Score = f64(math.Max(0, math.Min(1, cos)))
		r.Nanos, r.Runs = dur.Nanoseconds(), runs
	}
	return r
}

// cosineOfEmbeddings is a plain dot product, since all the embeddings are
// already L2-normalised. A zero vector (empty text, or nothing the vocabulary
// knows) yields 0.
func cosineOfEmbeddings(a, b []float32) float64 {
	var dot float64
	for i := range min(len(a), len(b)) {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}
