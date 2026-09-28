FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /similaritything .
RUN mkdir /models

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /similaritything /similaritything
# owned by the nonroot user so a fresh volume mounted here is writable
COPY --from=build --chown=65532:65532 /models /models
# embedding model cache (~1.5 GB for all models); mount a volume to keep it across restarts
ENV GO_POTION_HOME=/models
VOLUME /models
EXPOSE 8080
ENTRYPOINT ["/similaritything"]
