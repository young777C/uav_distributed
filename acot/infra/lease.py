"""Resource leases (framework §13.1): non-blocking file locks under runs/infra/leases/ so concurrent `acot eval` runs never
share a CARLA server, a policy GPU, a TM port block or a DAM4SAM instance."""
import fcntl
import json
import os
import time

from acot import ROOT

DIR = os.path.join(ROOT, "runs", "infra", "leases")


class Lease:
    def __init__(self, names, owner):
        self.names, self.owner, self.fds = list(names), owner, []

    def acquire(self):
        os.makedirs(DIR, exist_ok=True)
        try:
            for n in self.names:
                fd = os.open(os.path.join(DIR, n + ".lock"), os.O_RDWR | os.O_CREAT, 0o664)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    raise RuntimeError(f"resource {n} is leased by another acot run "
                                       f"({open(os.path.join(DIR, n + '.lock')).read().strip() or '?'})")
                os.ftruncate(fd, 0)
                os.write(fd, json.dumps({"owner": self.owner, "pid": os.getpid(), "t": time.time()}).encode())
                self.fds.append(fd)
        except Exception:
            self.release()
            raise
        return self

    def release(self):
        while self.fds:
            fd = self.fds.pop()
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
