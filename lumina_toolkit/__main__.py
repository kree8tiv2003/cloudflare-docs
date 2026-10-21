import argparse, sys
from PIL import Image
from . import APP_NAME, list_features, get_feature
from .backends import BackendMissing


def main(argv=None):
    p = argparse.ArgumentParser(prog="lumina", description=APP_NAME)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for key in ("enhance", "upscale", "remove_bg", "whiten", "colorize", "old_photo_repair"):
        s = sub.add_parser(key); s.add_argument("input"); s.add_argument("output")
    a = p.parse_args(argv)
    if a.cmd == "list":
        for f in list_features():
            print(f"{f.key:18} {f.title:30} {'[needs backend: '+f.needs_backend+']' if f.needs_backend else ''}")
        return 0
    try:
        get_feature(a.cmd).run(Image.open(a.input)).save(a.output)
    except BackendMissing as e:
        print(e, file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
