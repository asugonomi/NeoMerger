"""Lightweight, link-aware file catalog. Never opens model contents."""
import hashlib
import logging
import os
import threading
from collections import defaultdict

log = logging.getLogger("NeoMerger")


def canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def identity(path):
    st = os.stat(path)
    return (st.st_dev, st.st_ino) if st.st_ino else canonical(path)


def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return canonical(a) == canonical(b)


def walk_models(roots, extensions):
    seen = set()
    for root in roots:
        pending = [os.path.abspath(root)]
        while pending:
            directory = pending.pop()
            try:
                key = identity(directory)
                if key in seen:
                    continue
                seen.add(key)
                with os.scandir(directory) as entries:
                    entries = sorted(entries, key=lambda e: e.name.casefold())
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=True):
                            pending.append(entry.path)
                        elif entry.is_file(follow_symlinks=True):
                            if os.path.splitext(entry.name)[1].lower() in extensions:
                                yield entry.path
                        else:
                            log.warning("Unavailable/broken link: %s", entry.path)
                    except OSError as exc:
                        log.warning("Cannot inspect %s: %s", entry.path, exc)
            except OSError as exc:
                log.warning("Cannot scan %s: %s", directory, exc)


class Catalog:
    def __init__(self, roots, registry, extensions, kind):
        self.roots = roots
        self.registry = registry
        self.extensions = extensions
        self.kind = kind
        self.entries = None
        self.aliases = {}
        self.lock = threading.RLock()

    def refresh(self):
        with self.lock:
            roots = list(dict.fromkeys(os.path.abspath(p) for p in self.roots() if p))
            candidates = []
            try:
                for item in list(self.registry()):
                    path = getattr(item, "filename", None)
                    if path:
                        aliases = [getattr(item, k, "") for k in ("name", "title", "model_name")]
                        candidates.append((path, aliases))
            except Exception as exc:
                log.warning("Model registry unavailable; using directory scan: %s", exc)
            candidates.extend((p, []) for p in walk_models(roots, self.extensions))
            entries, aliases, physical = {}, defaultdict(set), {}
            for path, old_names in candidates:
                path = os.path.abspath(path)
                if os.path.splitext(path)[1].lower() not in self.extensions:
                    continue
                if self.kind == "checkpoint" and path.lower().endswith((".vae.ckpt", ".vae.safetensors")):
                    continue
                try:
                    if not os.path.isfile(path):
                        log.warning("Registered file missing: %s", path)
                        continue
                    key = identity(path)
                    real = canonical(path)
                except OSError as exc:
                    log.warning("Cannot inspect %s: %s", path, exc)
                    continue
                # Root slot + relative path is portable; no machine-specific absolute path.
                label = None
                for index, root in enumerate(roots):
                    try:
                        if os.path.commonpath([os.path.normcase(path), os.path.normcase(root)]) == os.path.normcase(root):
                            label = f"root{index + 1}/{os.path.relpath(path, root).replace(os.sep, '/')}"
                            break
                    except ValueError:
                        pass
                if label is None:
                    # Registry-only paths use their parent directory name; collisions fail closed.
                    label = "registry/" + os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
                    label += " [" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:16] + "]"
                token = physical.get(key)
                if token is None:
                    token = label
                    if token in entries:
                        # Do not silently replace a distinct registry-only model.
                        token = label + " [" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:16] + "]"
                    entries[token] = (path, real, key)
                    physical[key] = token
                for alias in [label, path, real, os.path.basename(path), *old_names]:
                    if alias:
                        aliases[str(alias)].add(token)
            self.entries, self.aliases = entries, aliases
            return self.choices()

    def choices(self):
        with self.lock:
            if self.entries is None:
                self.refresh()
            return sorted(self.entries, key=str.casefold)

    def select(self, name):
        with self.lock:
            self.choices()
            if not name:
                raise ValueError("Select a model first.")
            matches = self.aliases.get(str(name), set())
            if name in self.entries:
                matches = matches | {name}
            if len(matches) > 1:
                raise ValueError(f"Ambiguous model '{name}'. Select its folder-qualified entry again.")
            if not matches:
                raise ValueError(f"Model '{name}' is unavailable. Refresh and select it again.")
            token = next(iter(matches))
            path, real, key = self.entries[token]
            try:
                valid = os.path.isfile(path) and canonical(path) == real and identity(path) == key
            except OSError:
                valid = False
            if not valid:
                raise ValueError(f"Model '{name}' was removed or its link changed. Refresh and select it again.")
            return token

    def resolve(self, name):
        with self.lock:
            token = self.select(name)
            return self.entries[token][1]
