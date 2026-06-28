"""Small standard-library HTTP server for testing model-response forwarding."""

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class ResponseReceiver(BaseHTTPRequestHandler):
    output_path: Path | None = None
    max_body_bytes: int = 1024 * 1024

    def do_POST(self) -> None:
        if self.path != "/response":
            self.send_error(404, "Use POST /response")
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None or not content_length.isdigit():
            self.send_error(411, "Content-Length is required")
            return

        body_length = int(content_length)
        if body_length > self.max_body_bytes:
            self.send_error(413, "Response body is too large")
            return

        body = self.rfile.read(body_length)
        text = body.decode("utf-8", errors="replace")
        dataset = self.headers.get("X-Dataset", "unknown")
        elapsed = self.headers.get("X-Inference-Seconds", "unknown")
        received_at = datetime.now().astimezone().isoformat(timespec="seconds")
        record = (
            f"[{received_at}] dataset={dataset} inference_seconds={elapsed}\n"
            f"{text}\n"
        )

        print(f"\nReceived model response:\n{record}", flush=True)
        if self.output_path is not None:
            with self.output_path.open("a", encoding="utf-8") as output_file:
                output_file.write(record + "\n")

        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive and print forwarded Image RAG model responses."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional UTF-8 text file to append received responses to.",
    )
    parser.add_argument("--max-body-mb", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.max_body_mb < 1:
        raise SystemExit("--max-body-mb must be at least 1.")

    ResponseReceiver.output_path = args.output
    ResponseReceiver.max_body_bytes = args.max_body_mb * 1024 * 1024
    server = HTTPServer((args.host, args.port), ResponseReceiver)
    print(
        f"Listening for model responses on http://{args.host}:{args.port}/response",
        flush=True,
    )
    if args.output:
        print(f"Appending received text to {args.output}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReceiver stopped.", flush=True)
    finally:
        server.server_close()
