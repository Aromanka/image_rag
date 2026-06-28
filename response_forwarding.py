"""Send completed model outputs to a configured HTTP receiver."""

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def forward_text_response(
    target_url: str,
    output: str,
    *,
    dataset: str,
    inference_seconds: float,
    timeout_seconds: float,
) -> bool:
    """POST one model response as UTF-8 plain text without raising to callers."""
    request = Request(
        target_url,
        data=output.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "X-Dataset": dataset,
            "X-Inference-Seconds": f"{inference_seconds:.3f}",
            "X-Source": "image-rag",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.status
        print(
            f"Response forwarded: url={target_url} status={status_code}",
            flush=True,
        )
        return 200 <= status_code < 300
    except HTTPError as exc:
        print(
            f"Response forwarding failed: url={target_url} "
            f"status={exc.code} reason={exc.reason}",
            flush=True,
        )
    except (URLError, OSError, ValueError) as exc:
        print(f"Response forwarding failed: url={target_url} error={exc}", flush=True)
    return False
