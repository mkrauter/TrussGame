"""Why do two of the three recovered SavedModels fail to parse?

    python training/diagnose_savedmodels.py

Distinguishes the possibilities that matter. If a binary protobuf was ever read
as text and written back, invalid byte sequences become U+FFFD (ef bf bd) and
the file is unrecoverable -- the original bytes are simply gone. If instead it
was opened in text mode on Windows, lone 0x0a bytes became 0x0d0a, which is
reversible. Anything else and the weights may still be readable straight out of
the checkpoint, since `variables/` is a separate file from the graph.
"""
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent / 'recovered'
NAMES = ('retrained_200', 'retrained_250', 'truss_game_AI_model')
FILES = ('saved_model.pb', 'keras_metadata.pb',
         'variables/variables.index', 'variables/variables.data-00000-of-00001')


def main():
    print(f'{"model":<22}{"file":<40}{"bytes":>10}{"U+FFFD":>9}{"CRLF":>7}')
    print('-' * 88)
    for name in NAMES:
        for relative in FILES:
            path = BASE / name / relative
            if not path.exists():
                print(f'{name:<22}{relative:<40}{"missing":>10}')
                continue
            blob = path.read_bytes()
            replacement = blob.count(b'\xef\xbf\xbd')
            crlf = blob.count(b'\r\n')
            print(f'{name:<22}{relative:<40}{len(blob):>10,}{replacement:>9}{crlf:>7}')
        print()

    print('Where the parse actually gives up:')
    for name in NAMES:
        blob = (BASE / name / 'saved_model.pb').read_bytes()
        try:
            from tensorflow.core.protobuf import saved_model_pb2
            saved_model_pb2.SavedModel().ParseFromString(blob)
            print(f'  {name:<22} parses cleanly')
        except ImportError:
            print('  (tensorflow not importable here -- run with tfvenv)')
            return
        except Exception:
            # Binary search for the longest prefix that still parses: that is
            # where the damage starts, and tells us whether it is one bad spot
            # or the whole file.
            low, high = 0, len(blob)
            while low < high:
                mid = (low + high + 1) // 2
                try:
                    saved_model_pb2.SavedModel().ParseFromString(blob[:mid])
                    low = mid
                except Exception:
                    high = mid - 1
            print(f'  {name:<22} parses up to byte {low:,} of {len(blob):,} '
                  f'({low / len(blob) * 100:.1f}%)')


if __name__ == '__main__':
    main()
