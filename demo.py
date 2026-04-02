import argparse

from magic_hand import run_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Demo-only launcher for hand particles (no webcam).")
    parser.add_argument("--width", type=int, default=960, help="Output window width")
    parser.add_argument("--height", type=int, default=540, help="Output window height")
    parser.add_argument("--particles", type=int, default=3000, help="Number of particles")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rw = max(400, int(args.width * 0.62))
    rh = max(225, int(args.height * 0.62))
    run_demo(width=args.width, height=args.height, particles=args.particles, rw=rw, rh=rh)
