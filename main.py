"""
People tracking with persistent identity (torchreid / OSNet) and door entry/exit events.

Layers:
  1. YOLO + BoT-SORT -> boxes with a tracker id (temporary: changes after a long occlusion)
  2. Embedder        -> appearance vector of a person crop (OSNet)
  3. Gallery         -> remembers every person (global id "P1", "P2", ...), several views each
  4. DoorLine        -> ENTER / EXIT events from a virtual line

Re-entry: a new id is provisional ("P5?") for a few seconds. If later crops match an older person,
the two ids are merged. A person seen while a track already existed cannot be that track's returnee.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ============================== CONFIG ==============================
VIDEO_PATH = "testing/video1.mp4"
# VIDEO_PATH = 0
MODEL_PATH = "./models/yolo26l.pt"

TORCHREID_WEIGHTS = "./models/osnet_ain_x1_0_imagenet.pth"   # Re-ID-trained OSNet checkpoint
TORCHREID_MODEL = "osnet_ain_x1_0"                         # must match the checkpoint architecture

FRAME_WIDTH = 960          # resize keeping aspect ratio
DET_CONF = 0.4

# Door line as fractions of the frame: ((x1, y1), (x2, y2)). Edit for your video.
DOOR_LINE = ((0.0, 0.55), (1.0, 0.55))
INSIDE_IS_POSITIVE_SIDE = True   # for a left-to-right line, the positive side is BELOW it
DOOR_BAND_PX = 12                # dead band around the line (prevents flicker counts)

# Re-ID: TUNE these using the [ID] lines printed in the console
MATCH_THRESHOLD = 0.70     # similarity needed to say "same person"
MATCH_MARGIN = 0.02        # best match must beat the 2nd best by this much
MIN_SAMPLES = 3            # good crops needed before a track gets an identity
MAX_SAMPLES = 8
EMBED_EVERY = 5            # frames between embeddings per track
BANK_SIZE = 12             # appearance vectors kept per person (different views)
BANK_DIVERSITY = 0.93      # only store a new vector if it differs enough from stored ones
BANK_ADD_MIN_SIM = 0.45    # weak sanity check before adding a crop to a person's bank
RECHECK_S = 8.0            # a NEW id stays provisional this long and may merge into an older id
COEXIST_GRACE_S = 1.0      # a person seen later than (track birth + this) can't be the same person
RETENTION_S = 3600         # forget people not seen for this long (video seconds)
LOST_FRAMES = 30           # drop a track's state after this many unseen frames

# What counts as a "good" crop for Re-ID
MIN_CROP_H, MIN_CROP_W = 80, 30
MIN_CROP_CONF = 0.5
MAX_OVERLAP_IOU = 0.25     # skip crops where another person overlaps heavily
DEBUG_SIM = True           # print similarity scores when an identity is created


# ============================== HELPERS ==============================
def l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ============================== EMBEDDER ==============================
class Embedder:
    """Turns a person crop into an appearance vector (unit length) using torchreid OSNet."""

    def __init__(self):
        import torch
        from torchreid.reid.utils import FeatureExtractor

        if not os.path.isfile(TORCHREID_WEIGHTS):
            raise FileNotFoundError(f"Re-ID weights not found: {TORCHREID_WEIGHTS}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor(
            model_name=TORCHREID_MODEL, model_path=TORCHREID_WEIGHTS, device=device
        )
        print(f"[Embedder] torchreid {TORCHREID_MODEL} on {device}")

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        feat = self.extractor([rgb])[0].detach().cpu().numpy()
        return l2(feat.reshape(-1))


# ============================== GALLERY ==============================
@dataclass
class Person:
    gid: int
    bank: list                 # several unit vectors: different views of the same person
    first_seen: float
    last_seen: float
    visits: int = 0            # number of ENTER events
    inside: Optional[bool] = None

    def score(self, emb: np.ndarray) -> float:
        """Similarity of `emb` to this person = best match among the stored views."""
        return max(float(np.dot(emb, b)) for b in self.bank)


class Gallery:
    """Long-term memory of everyone seen."""

    def __init__(self, threshold: float, margin: float, retention_s: float):
        self.persons: dict[int, Person] = {}
        self.threshold = threshold
        self.margin = margin
        self.retention_s = retention_s
        self._next = 1

    @staticmethod
    def _prune(p: Person):
        """Keep the bank small by dropping the most redundant vector first."""
        while len(p.bank) > BANK_SIZE:
            redundancy = [max(float(np.dot(p.bank[i], p.bank[j])) for j in range(len(p.bank)) if j != i)
                          for i in range(len(p.bank))]
            p.bank.pop(int(np.argmax(redundancy)))

    def _bank_add(self, p: Person, emb: np.ndarray):
        if max(float(np.dot(emb, b)) for b in p.bank) < BANK_DIVERSITY:
            p.bank.append(emb)
            self._prune(p)

    def create(self, embs: list, now: float) -> int:
        gid = self._next
        self._next += 1
        p = Person(gid, [], now, now)
        for e in embs:
            if not p.bank:
                p.bank.append(e)
            else:
                self._bank_add(p, e)
        self.persons[gid] = p
        return gid

    def match(self, emb: np.ndarray, exclude: set, now: float, seen_before: Optional[float] = None):
        """Return (gid or None, best_score, second_score).

        exclude      : ids that must not be chosen (people currently on screen)
        seen_before  : candidates last seen AFTER this time are skipped (they coexisted with the
                       query track, so they cannot be the same person)
        """
        cands = [p for p in self.persons.values()
                 if p.gid not in exclude
                 and now - p.last_seen <= self.retention_s
                 and (seen_before is None or p.last_seen <= seen_before)]
        if not cands:
            return None, -1.0, -1.0
        scores = sorted(((p.score(emb), p.gid) for p in cands), reverse=True)
        best, gid = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        if best >= self.threshold and (best - second) >= self.margin:
            return gid, best, second
        return None, best, second

    def add(self, gid: int, emb: np.ndarray, now: float):
        """Add a new view of a person we are already tracking."""
        p = self.persons[gid]
        p.last_seen = now
        if p.score(emb) >= BANK_ADD_MIN_SIM:
            self._bank_add(p, emb)

    def merge(self, src: int, dst: int):
        """Fold person `src` into `dst` (they turned out to be the same person)."""
        a, b = self.persons[src], self.persons[dst]
        for e in a.bank:
            self._bank_add(b, e)
        b.visits += a.visits
        b.first_seen = min(a.first_seen, b.first_seen)
        b.last_seen = max(a.last_seen, b.last_seen)
        if a.inside is not None:
            b.inside = a.inside
        del self.persons[src]

    def touch(self, gid: int, now: float):
        self.persons[gid].last_seen = now

    def expire(self, now: float):
        for gid in [g for g, p in self.persons.items() if now - p.last_seen > self.retention_s]:
            del self.persons[gid]


# ============================== DOOR ==============================
class DoorLine:
    """Virtual door line in pixel space with a dead band, evaluated on the person's feet."""

    def __init__(self, line_frac, width, height, inside_is_positive=True, band_px=12):
        (x1, y1), (x2, y2) = line_frac
        self.p1 = np.array([x1 * width, y1 * height], dtype=np.float64)
        self.p2 = np.array([x2 * width, y2 * height], dtype=np.float64)
        self.d = self.p2 - self.p1
        self.length = float(np.linalg.norm(self.d))
        self.inside_sign = 1 if inside_is_positive else -1
        self.band = band_px

    def side_of(self, pt) -> int:
        """+1 / -1 = clearly on one side, 0 = inside the dead band or beside the door segment."""
        v = np.asarray(pt, dtype=np.float64) - self.p1
        t = float(np.dot(v, self.d) / (self.length ** 2))
        if t < -0.05 or t > 1.05:          # not in front of the door segment
            return 0
        dist = float((self.d[0] * v[1] - self.d[1] * v[0]) / self.length)
        if dist > self.band:
            return 1
        if dist < -self.band:
            return -1
        return 0


# ============================== TRACK STATE + IDENTITY ==============================
@dataclass
class TrackState:
    born_time: float = 0.0
    gid: Optional[int] = None
    is_new: bool = False                  # identity was created for this track (provisional)
    assigned_frame: int = 0
    samples: list = field(default_factory=list)   # embeddings collected before an identity exists
    recent: list = field(default_factory=list)    # latest embeddings after assignment
    last_frame: int = 0
    last_embed_frame: int = -10_000
    side: Optional[int] = None
    inside_state: Optional[bool] = None   # last door event seen on this track


def is_provisional(ts: TrackState, frame_idx: int, recheck_frames: int) -> bool:
    return ts.gid is not None and ts.is_new and frame_idx - ts.assigned_frame <= recheck_frames


def update_identity(tid: int, emb: np.ndarray, tracks: dict, present: set, gallery: Gallery,
                    now: float, frame_idx: int, recheck_frames: int, log=print):
    """Feed one new embedding of track `tid`. Assigns, recognises or merges identities."""
    ts = tracks[tid]
    on_screen = {tracks[t].gid for t in present if t != tid and tracks[t].gid is not None}
    seen_before = ts.born_time + COEXIST_GRACE_S

    # ---- 1. no identity yet: collect crops, then match or create ----
    if ts.gid is None:
        ts.samples = (ts.samples + [emb])[-MAX_SAMPLES:]
        if len(ts.samples) < MIN_SAMPLES:
            return
        mean_emb = l2(np.mean(ts.samples, axis=0))
        gid, best, second = gallery.match(mean_emb, on_screen, now, seen_before)
        if gid is None:
            gid = gallery.create(ts.samples, now)
            ts.is_new = True
            if DEBUG_SIM:
                log(f"[ID] track {tid}: NEW person P{gid} (provisional; best={best:.2f}, 2nd={second:.2f})")
        else:
            for s in ts.samples:
                gallery.add(gid, s, now)
            log(f"[ID] track {tid}: RECOGNISED as P{gid} (score={best:.2f}, 2nd={second:.2f})")
        ts.gid = gid
        ts.assigned_frame = frame_idx
        ts.recent = list(ts.samples)
        ts.samples = []
        if ts.inside_state is not None:
            gallery.persons[gid].inside = ts.inside_state
        return

    # ---- 2. has identity: grow the bank, and re-check provisional identities ----
    ts.recent = (ts.recent + [emb])[-MAX_SAMPLES:]
    gallery.add(ts.gid, emb, now)

    if is_provisional(ts, frame_idx, recheck_frames) and len(ts.recent) >= MIN_SAMPLES:
        mean_emb = l2(np.mean(ts.recent, axis=0))
        gid, best, second = gallery.match(mean_emb, on_screen | {ts.gid}, now, seen_before)
        if gid is not None:
            old = ts.gid
            gallery.merge(old, gid)
            ts.gid, ts.is_new = gid, False
            log(f"[ID] track {tid}: P{old} was actually P{gid} -> MERGED "
                f"(score={best:.2f}, 2nd={second:.2f}); P{gid} now has {gallery.persons[gid].visits} visit(s)")
    elif ts.is_new and not is_provisional(ts, frame_idx, recheck_frames):
        ts.is_new = False                 # window over: identity is final


def good_crop(box, conf, others, frame_shape) -> bool:
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    if conf < MIN_CROP_CONF:
        return False
    if (y2 - y1) < MIN_CROP_H or (x2 - x1) < MIN_CROP_W:
        return False
    if x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2:     # cut off by the frame edge
        return False
    return all(iou(box, o) <= MAX_OVERLAP_IOU for o in others)


def crop_person(frame, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * 0.05), int((y2 - y1) * 0.03)     # trim a little background
    return frame[max(0, y1 + my): y2 - my, max(0, x1 + mx): x2 - mx]


def color_for(gid: Optional[int]):
    if gid is None:
        return (160, 160, 160)
    rng = np.random.default_rng(gid * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


# ============================== MAIN ==============================
def main():
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    embedder = Embedder()
    gallery = Gallery(MATCH_THRESHOLD, MATCH_MARGIN, RETENTION_S)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    recheck_frames = int(RECHECK_S * fps)

    tracks: dict[int, TrackState] = {}
    door: Optional[DoorLine] = None
    entries = exits = 0
    frame_idx = 0

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, FRAME_WIDTH)
        H, W = frame.shape[:2]
        now = frame_idx / fps            # video time; for a live camera use time.time()

        if door is None:
            door = DoorLine(DOOR_LINE, W, H, INSIDE_IS_POSITIVE_SIDE, DOOR_BAND_PX)

        result = model.track(frame, tracker="botsort.yaml", persist=True,
                             conf=DET_CONF, classes=[0], verbose=False)[0]

        dets = []   # (tracker_id, int box, conf)
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            dets = [(tid, b, float(c)) for tid, b, c in zip(ids, xyxy, confs)]

        # pass 1: mark every visible track as present (needed for the "one id per person" rule)
        present = set()
        for tid, _, _ in dets:
            if tid not in tracks:
                tracks[tid] = TrackState(born_time=now)
            tracks[tid].last_frame = frame_idx
            present.add(tid)

        # pass 2: identity + door logic
        for tid, box, conf in dets:
            ts = tracks[tid]
            others = [b for t, b, _ in dets if t != tid]

            # --- identity ---
            if frame_idx - ts.last_embed_frame >= EMBED_EVERY and good_crop(box, conf, others, frame.shape):
                crop = crop_person(frame, box)
                if crop.size > 0:
                    ts.last_embed_frame = frame_idx
                    update_identity(tid, embedder.embed(crop), tracks, present, gallery,
                                    now, frame_idx, recheck_frames)

            if ts.gid is not None and ts.gid in gallery.persons:
                gallery.touch(ts.gid, now)

            # --- door crossing (feet point) ---
            foot = ((box[0] + box[2]) // 2, box[3])
            side = door.side_of(foot)
            if side != 0:
                if ts.side is not None and side != ts.side:
                    is_enter = side == door.inside_sign
                    entries += is_enter
                    exits += not is_enter
                    ts.inside_state = is_enter
                    who = f"P{ts.gid}" if ts.gid is not None else f"track {tid} (id pending)"
                    tag = ""
                    if ts.gid is not None:
                        p = gallery.persons[ts.gid]
                        p.inside = is_enter
                        if is_enter:
                            p.visits += 1
                            tag = f" - visit #{p.visits}" + (" (RE-ENTRY)" if p.visits > 1 else "")
                    print(f"[DOOR] {who} {'ENTER' if is_enter else 'EXIT'}{tag}")
                ts.side = side

            # --- draw ---
            color = color_for(ts.gid)
            if ts.gid is None:
                label = "P? (learning)"
            else:
                label = f"P{ts.gid}" + ("?" if is_provisional(ts, frame_idx, recheck_frames) else "")
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{label} t{tid}", (box[0] + 4, box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, foot, 4, color, -1)

        # --- housekeeping ---
        for tid in [t for t, s in tracks.items() if frame_idx - s.last_frame > LOST_FRAMES]:
            del tracks[tid]
        gallery.expire(now)

        # --- HUD ---
        cv2.line(frame, tuple(door.p1.astype(int)), tuple(door.p2.astype(int)), (0, 255, 255), 2)
        fps_now = 1.0 / max(time.time() - t0, 1e-6)
        hud = f"inside: {entries - exits}  in: {entries}  out: {exits}  known people: {len(gallery.persons)}  fps: {fps_now:.0f}"
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("detection", frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()