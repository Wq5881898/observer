"""Both bootloaders use this entry; no arguments means the desktop GUI."""
from observer.frozen_entry import main

if __name__ == '__main__':
    raise SystemExit(main())
