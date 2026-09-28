// Command similaritything serves similaritything.com: a page that compares
// two pieces of text with a range of string-similarity, phonetic and
// static-embedding algorithms, and reports each result and its runtime.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"flag"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
	"unicode/utf8"
)

//go:embed web
var webFS embed.FS

type compareRequest struct {
	A          string `json:"a"`
	B          string `json:"b"`
	IgnoreCase bool   `json:"ignoreCase"`
}

type compareResponse struct {
	Results    []Result      `json:"results"`
	TotalNanos int64         `json:"totalNanos"`
	Models     []ModelStatus `json:"models"`
}

func main() {
	addr := flag.String("addr", defaultAddr(), "listen address")
	models := flag.String("models", envOr("POTION_MODELS", "all"),
		`comma separated embedding models to load (e.g. "BASE2M,BASE8M"), "all" or "none"`)
	maxLen := flag.Int("max-len", 2000, "maximum characters per text box")
	flag.Parse()

	emb, err := newEmbedder(*models)
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go emb.loadAll(ctx)

	static, _ := fs.Sub(webFS, "web")
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServerFS(static))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("GET /api/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, emb.status())
	})
	// decode reads a JSON request holding texts a and b, replying with an
	// error if it is malformed or either text is too long.
	decode := func(w http.ResponseWriter, r *http.Request, req any, a, b *string) bool {
		r.Body = http.MaxBytesReader(w, r.Body, int64(*maxLen)*8+1024)
		if err := json.NewDecoder(r.Body).Decode(req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request"})
			return false
		}
		if utf8.RuneCountInString(*a) > *maxLen || utf8.RuneCountInString(*b) > *maxLen {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{"error": "text too long", "maxLen": *maxLen})
			return false
		}
		return true
	}
	mux.HandleFunc("POST /api/compare", func(w http.ResponseWriter, r *http.Request) {
		var req compareRequest
		if !decode(w, r, &req, &req.A, &req.B) {
			return
		}
		start := time.Now()
		results := compareStrings(req.A, req.B, req.IgnoreCase)
		results = append(results, emb.compare(req.A, req.B)...)
		// List the phonetic group last, after the embeddings.
		var rest, phonetic []Result
		for _, r := range results {
			if r.Group == groupPhonetic {
				phonetic = append(phonetic, r)
			} else {
				rest = append(rest, r)
			}
		}
		results = append(rest, phonetic...)
		writeJSON(w, http.StatusOK, compareResponse{
			Results:    results,
			TotalNanos: time.Since(start).Nanoseconds(),
			Models:     emb.status(),
		})
	})

	srv := &http.Server{
		Addr:              *addr,
		Handler:           securityHeaders(mux),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      30 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.Shutdown(shutdown)
	}()

	log.Printf("similaritything listening on %s", *addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "strict-origin-when-cross-origin")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Content-Security-Policy",
			"default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "+
				"font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; img-src 'self' data:")
		next.ServeHTTP(w, r)
	})
}

func defaultAddr() string {
	if p := os.Getenv("PORT"); p != "" {
		return ":" + p
	}
	return ":8080"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
