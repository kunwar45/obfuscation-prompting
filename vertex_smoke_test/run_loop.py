import os
import time


def main() -> None:
    steps = int(os.getenv("SMOKE_STEPS", "5"))
    sleep_seconds = float(os.getenv("SMOKE_SLEEP_SECONDS", "2"))

    print("VERTEX_SMOKE_TEST_START", flush=True)
    print(
        f"Configured steps={steps} sleep_seconds={sleep_seconds}",
        flush=True,
    )

    for step in range(1, steps + 1):
        print(f"VERTEX_LOOP_STEP_{step}", flush=True)
        time.sleep(sleep_seconds)

    print("VERTEX_SMOKE_TEST_SUCCESS", flush=True)


if __name__ == "__main__":
    main()
