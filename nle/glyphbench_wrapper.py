"""GlyphBench-facing compatibility wrapper for full NetHack.

This module stays inside the NLE fork so downstream benchmark code can depend
on a small, stable Python surface instead of reaching into NLE internals.  It
keeps the observation glyph-native: the map is rendered from NLE's ``chars``
array, while status, inventory, and message/menu text are decoded from the
standard NLE observation tensors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from nle import nethack
from nle.env.tasks import NetHackScore

ACTION_KEYS: tuple[tuple[str, int], ...] = (
    ("MOVE_N", nethack.CompassDirection.N),
    ("MOVE_E", nethack.CompassDirection.E),
    ("MOVE_S", nethack.CompassDirection.S),
    ("MOVE_W", nethack.CompassDirection.W),
    ("MOVE_NE", nethack.CompassDirection.NE),
    ("MOVE_SE", nethack.CompassDirection.SE),
    ("MOVE_SW", nethack.CompassDirection.SW),
    ("MOVE_NW", nethack.CompassDirection.NW),
    ("RUN_N", nethack.CompassDirectionLonger.N),
    ("RUN_E", nethack.CompassDirectionLonger.E),
    ("RUN_S", nethack.CompassDirectionLonger.S),
    ("RUN_W", nethack.CompassDirectionLonger.W),
    ("RUN_NE", nethack.CompassDirectionLonger.NE),
    ("RUN_SE", nethack.CompassDirectionLonger.SE),
    ("RUN_SW", nethack.CompassDirectionLonger.SW),
    ("RUN_NW", nethack.CompassDirectionLonger.NW),
    ("GO_UP", nethack.MiscDirection.UP),
    ("GO_DOWN", nethack.MiscDirection.DOWN),
    ("WAIT", nethack.MiscDirection.WAIT),
    ("MORE", nethack.MiscAction.MORE),
    ("EXTCMD", nethack.Command.EXTCMD),
    ("EXTLIST", nethack.Command.EXTLIST),
    ("ADJUST", nethack.Command.ADJUST),
    ("ANNOTATE", nethack.Command.ANNOTATE),
    ("APPLY", nethack.Command.APPLY),
    ("ATTRIBUTES", nethack.Command.ATTRIBUTES),
    ("AUTOPICKUP", nethack.Command.AUTOPICKUP),
    ("CALL", nethack.Command.CALL),
    ("CAST", nethack.Command.CAST),
    ("CHAT", nethack.Command.CHAT),
    ("CLOSE", nethack.Command.CLOSE),
    ("CONDUCT", nethack.Command.CONDUCT),
    ("DIP", nethack.Command.DIP),
    ("DROP", nethack.Command.DROP),
    ("DROPTYPE", nethack.Command.DROPTYPE),
    ("EAT", nethack.Command.EAT),
    ("ENGRAVE", nethack.Command.ENGRAVE),
    ("ENHANCE", nethack.Command.ENHANCE),
    ("ESC", nethack.Command.ESC),
    ("FIGHT", nethack.Command.FIGHT),
    ("FIRE", nethack.Command.FIRE),
    ("FORCE", nethack.Command.FORCE),
    ("GLANCE", nethack.Command.GLANCE),
    ("HISTORY", nethack.Command.HISTORY),
    ("INVENTORY", nethack.Command.INVENTORY),
    ("INVENTTYPE", nethack.Command.INVENTTYPE),
    ("INVOKE", nethack.Command.INVOKE),
    ("JUMP", nethack.Command.JUMP),
    ("KICK", nethack.Command.KICK),
    ("KNOWN", nethack.Command.KNOWN),
    ("KNOWNCLASS", nethack.Command.KNOWNCLASS),
    ("LOOK", nethack.Command.LOOK),
    ("LOOT", nethack.Command.LOOT),
    ("MONSTER", nethack.Command.MONSTER),
    ("MOVE", nethack.Command.MOVE),
    ("MOVEFAR", nethack.Command.MOVEFAR),
    ("OFFER", nethack.Command.OFFER),
    ("OPEN", nethack.Command.OPEN),
    ("OPTIONS", nethack.Command.OPTIONS),
    ("OVERVIEW", nethack.Command.OVERVIEW),
    ("PAY", nethack.Command.PAY),
    ("PICKUP", nethack.Command.PICKUP),
    ("PRAY", nethack.Command.PRAY),
    ("PUTON", nethack.Command.PUTON),
    ("QUAFF", nethack.Command.QUAFF),
    ("QUIT", nethack.Command.QUIT),
    ("QUIVER", nethack.Command.QUIVER),
    ("READ", nethack.Command.READ),
    ("REDRAW", nethack.Command.REDRAW),
    ("REMOVE", nethack.Command.REMOVE),
    ("RIDE", nethack.Command.RIDE),
    ("RUB", nethack.Command.RUB),
    ("RUSH", nethack.Command.RUSH),
    ("RUSH2", nethack.Command.RUSH2),
    ("SAVE", nethack.Command.SAVE),
    ("SEARCH", nethack.Command.SEARCH),
    ("SEEALL", nethack.Command.SEEALL),
    ("SEEAMULET", nethack.Command.SEEAMULET),
    ("SEEARMOR", nethack.Command.SEEARMOR),
    ("SEEGOLD", nethack.Command.SEEGOLD),
    ("SEERINGS", nethack.Command.SEERINGS),
    ("SEESPELLS", nethack.Command.SEESPELLS),
    ("SEETOOLS", nethack.Command.SEETOOLS),
    ("SEETRAP", nethack.Command.SEETRAP),
    ("SEEWEAPON", nethack.Command.SEEWEAPON),
    ("SHELL", nethack.Command.SHELL),
    ("SIT", nethack.Command.SIT),
    ("SWAP", nethack.Command.SWAP),
    ("TAKEOFF", nethack.Command.TAKEOFF),
    ("TAKEOFFALL", nethack.Command.TAKEOFFALL),
    ("TELEPORT", nethack.Command.TELEPORT),
    ("THROW", nethack.Command.THROW),
    ("TIP", nethack.Command.TIP),
    ("TRAVEL", nethack.Command.TRAVEL),
    ("TURN", nethack.Command.TURN),
    ("TWOWEAPON", nethack.Command.TWOWEAPON),
    ("UNTRAP", nethack.Command.UNTRAP),
    ("VERSION", nethack.Command.VERSION),
    ("VERSIONSHORT", nethack.Command.VERSIONSHORT),
    ("WEAR", nethack.Command.WEAR),
    ("WHATDOES", nethack.Command.WHATDOES),
    ("WHATIS", nethack.Command.WHATIS),
    ("WIELD", nethack.Command.WIELD),
    ("WIPE", nethack.Command.WIPE),
    ("ZAP", nethack.Command.ZAP),
    ("TEXT_SPACE", nethack.TextCharacters.SPACE),
    ("TEXT_APOSTROPHE", nethack.TextCharacters.APOS),
    ("TEXT_QUOTE", nethack.TextCharacters.QUOTE),
    ("TEXT_PLUS", nethack.TextCharacters.PLUS),
    ("TEXT_MINUS", nethack.TextCharacters.MINUS),
    ("TEXT_DOLLAR", nethack.TextCharacters.DOLLAR),
    ("TEXT_0", nethack.TextCharacters.NUM_0),
    ("TEXT_1", nethack.TextCharacters.NUM_1),
    ("TEXT_2", nethack.TextCharacters.NUM_2),
    ("TEXT_3", nethack.TextCharacters.NUM_3),
    ("TEXT_4", nethack.TextCharacters.NUM_4),
    ("TEXT_5", nethack.TextCharacters.NUM_5),
    ("TEXT_6", nethack.TextCharacters.NUM_6),
    ("TEXT_7", nethack.TextCharacters.NUM_7),
    ("TEXT_8", nethack.TextCharacters.NUM_8),
    ("TEXT_9", nethack.TextCharacters.NUM_9),
)

ACTION_NAMES: tuple[str, ...] = tuple(name for name, _ in ACTION_KEYS)
ACTION_ENUMS: tuple[int, ...] = tuple(value for _, value in ACTION_KEYS)

OBSERVATION_KEYS: tuple[str, ...] = (
    "glyphs",
    "chars",
    "colors",
    "specials",
    "blstats",
    "message",
    "inv_glyphs",
    "inv_strs",
    "inv_letters",
    "inv_oclasses",
    "screen_descriptions",
    "tty_chars",
    "tty_colors",
    "tty_cursor",
    "misc",
)

SCORE_CAP = 10_000.0
GRID_CROP_PADDING = 1
_TOP_TEN_SCORE_RE = re.compile(r"^\s*\d+\s+(\d+)\s+Agent-", re.MULTILINE)

_HUNGER = {
    0: "Satiated",
    1: "Not Hungry",
    2: "Hungry",
    3: "Weak",
    4: "Fainting",
    5: "Fainted",
    6: "Starved",
}
_ENCUMBRANCE = {
    0: "Unencumbered",
    1: "Burdened",
    2: "Stressed",
    3: "Strained",
    4: "Overtaxed",
    5: "Overloaded",
}
_ALIGNMENT = {
    -128: "None",
    -1: "Chaotic",
    0: "Neutral",
    1: "Lawful",
}
_CONDITION_MASKS: tuple[tuple[int, str], ...] = (
    (0x00000001, "Stoned"),
    (0x00000002, "Slimed"),
    (0x00000004, "Strangled"),
    (0x00000008, "Food Poisoning"),
    (0x00000010, "Terminally Ill"),
    (0x00000020, "Blind"),
    (0x00000040, "Deaf"),
    (0x00000080, "Stunned"),
    (0x00000100, "Confused"),
    (0x00000200, "Hallucinating"),
    (0x00000400, "Levitating"),
    (0x00000800, "Flying"),
    (0x00001000, "Riding"),
)


@dataclass(frozen=True, slots=True)
class GlyphbenchNLEObservation:
    grid: str
    legend: str
    hud: str
    message: str
    score: int
    hp: int
    time: int


@dataclass(frozen=True, slots=True)
class GridViewport:
    y0: int
    x0: int
    y1: int
    x1: int
    full_height: int
    full_width: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


class GlyphbenchNetHack:
    """Small adapter around a seedable full-action NetHack environment."""

    def __init__(
        self,
        *,
        max_episode_steps: int = 100_000,
        character: str = "@",
        score_cap: float | None = None,
    ) -> None:
        del score_cap  # kept for compatibility with older callers
        self._env = NetHackScore(
            actions=ACTION_ENUMS,
            character=character,
            allow_all_yn_questions=True,
            allow_all_modes=True,
            penalty_mode="constant",
            penalty_step=0.0,
            penalty_time=0.0,
            max_episode_steps=max_episode_steps,
            observation_keys=OBSERVATION_KEYS,
            fix_moon_phase=True,
        )
        self._best_score = 0
        self._last_obs: dict[str, np.ndarray] | None = None

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_NAMES

    def reset(self, seed: int) -> tuple[GlyphbenchNLEObservation, dict[str, Any]]:
        rng = np.random.default_rng(int(seed))
        core = int(rng.integers(0, 2**31 - 1))
        disp = int(rng.integers(0, 2**31 - 1))
        lgen = int(rng.integers(0, 2**31 - 1))
        self._env.seed(core=core, disp=disp, reseed=False, lgen=lgen)
        obs, info = self._env.reset()
        self._last_obs = obs
        self._best_score = self._score_from_obs(obs)
        return self._render(obs), {
            **dict(info),
            "nle_core_seed": core,
            "nle_disp_seed": disp,
            "nle_level_seed": lgen,
        }

    def step(
        self, action_index: int
    ) -> tuple[GlyphbenchNLEObservation, float, bool, bool, dict[str, Any]]:
        obs, nle_task_reward, terminated, truncated, info = self._env.step(
            int(action_index)
        )
        self._last_obs = obs
        rendered = self._render(obs)
        raw_score = self._score_from_obs(obs)
        terminal_score = _terminal_score_from_message(rendered.message)
        if terminal_score is not None:
            raw_score = max(raw_score, terminal_score)
        previous_score = self._best_score
        self._best_score = max(self._best_score, raw_score)
        reward = float(self._best_score - previous_score)

        blstats = obs["blstats"]
        info = {
            **dict(info),
            "nle_raw_reward": float(nle_task_reward),
            "nle_task_reward": float(nle_task_reward),
            "raw_score_delta": reward,
            "score": int(self._best_score),
            "nethack_score": int(self._best_score),
            "nethack_blstats_score": int(blstats[nethack.NLE_BL_SCORE]),
            "nethack_terminal_score": terminal_score,
            "hp": int(blstats[nethack.NLE_BL_HP]),
            "time": int(blstats[nethack.NLE_BL_TIME]),
        }
        return rendered, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._env.close()

    def _score_from_obs(self, obs: dict[str, np.ndarray]) -> int:
        score = int(obs["blstats"][nethack.NLE_BL_SCORE])
        return max(score, 0)

    def _render(self, obs: dict[str, np.ndarray]) -> GlyphbenchNLEObservation:
        blstats = obs["blstats"]
        viewport = _grid_viewport(obs["chars"])
        return GlyphbenchNLEObservation(
            grid=_render_grid(obs["chars"], viewport),
            legend=_render_legend(
                obs["chars"], obs["screen_descriptions"], viewport
            ),
            hud=_render_hud(
                blstats, obs["inv_strs"], obs["inv_letters"], viewport, obs["chars"]
            ),
            message=_render_message(obs["message"], obs["tty_chars"]),
            score=int(blstats[nethack.NLE_BL_SCORE]),
            hp=int(blstats[nethack.NLE_BL_HP]),
            time=int(blstats[nethack.NLE_BL_TIME]),
        )


def _grid_viewport(chars: np.ndarray) -> GridViewport:
    height, width = int(chars.shape[0]), int(chars.shape[1])
    visible = (chars != 0) & (chars != ord(" "))
    ys, xs = np.nonzero(visible)
    if len(ys) == 0 or len(xs) == 0:
        return GridViewport(0, 0, min(height, 1), min(width, 1), height, width)

    y0 = max(0, int(ys.min()) - GRID_CROP_PADDING)
    y1 = min(height, int(ys.max()) + GRID_CROP_PADDING + 1)
    x0 = max(0, int(xs.min()) - GRID_CROP_PADDING)
    x1 = min(width, int(xs.max()) + GRID_CROP_PADDING + 1)
    return GridViewport(y0, x0, y1, x1, height, width)


def _render_grid(chars: np.ndarray, viewport: GridViewport) -> str:
    rows: list[str] = []
    for row in chars[viewport.y0 : viewport.y1, viewport.x0 : viewport.x1]:
        rows.append("".join(chr(int(c)) if int(c) != 0 else " " for c in row))
    return "\n".join(rows)


def _decode_c_string(row: np.ndarray) -> str:
    values = []
    for c in row.flat:
        i = int(c)
        if i == 0:
            break
        values.append(i)
    return bytes(values).decode("latin-1", errors="replace").strip()


def _render_legend(
    chars: np.ndarray,
    screen_descriptions: np.ndarray,
    viewport: GridViewport,
) -> str:
    meanings: dict[str, list[str]] = {}
    for y in range(viewport.y0, viewport.y1):
        for x in range(viewport.x0, viewport.x1):
            ch = chr(int(chars[y, x])) if int(chars[y, x]) != 0 else " "
            desc = _decode_c_string(screen_descriptions[y, x])
            if not desc:
                continue
            key = "space" if ch == " " else ch
            if desc not in meanings.setdefault(key, []):
                meanings[key].append(desc)

    if not meanings:
        return ""
    lines = []
    for key in sorted(meanings):
        joined = " / ".join(meanings[key][:4])
        if len(meanings[key]) > 4:
            joined += " / ..."
        symbol = "` ` (space)" if key == "space" else key
        lines.append(f"{symbol} — {joined}")
    return "\n".join(lines)


def _render_hud(
    blstats: np.ndarray,
    inv_strs: np.ndarray,
    inv_letters: np.ndarray,
    viewport: GridViewport,
    chars: np.ndarray,
) -> str:
    inventory = _render_inventory(inv_strs, inv_letters)
    conditions = [
        label for mask, label in _CONDITION_MASKS if int(blstats[25]) & mask
    ]
    condition = " ".join(conditions) if conditions else "None"
    grid_view = _render_grid_viewport(viewport, chars)
    return "\n".join(
        (
            grid_view,
            f"HP: {int(blstats[10])}/{int(blstats[11])}    "
            f"Pw: {int(blstats[14])}/{int(blstats[15])}    "
            f"AC: {int(blstats[16])}    "
            f"XP: {int(blstats[18])}/{int(blstats[19])}    "
            f"Score: {int(blstats[9])}    Gold: {int(blstats[13])}",
            f"Str: {int(blstats[2])}    Str125: {int(blstats[3])}    "
            f"Dex: {int(blstats[4])}    Con: {int(blstats[5])}    "
            f"Int: {int(blstats[6])}    Wis: {int(blstats[7])}    "
            f"Cha: {int(blstats[8])}",
            f"Dlvl: {int(blstats[12])}    "
            f"Dungeon: {int(blstats[23])}    Level: {int(blstats[24])}    "
            f"Position: {int(blstats[0])},{int(blstats[1])}    "
            f"Time: {int(blstats[20])}",
            f"Hunger: {_HUNGER.get(int(blstats[21]), str(int(blstats[21])))}    "
            f"Encumbrance: {_ENCUMBRANCE.get(int(blstats[22]), str(int(blstats[22])))}    "
            f"Alignment: {_ALIGNMENT.get(int(blstats[26]), str(int(blstats[26])))}    "
            f"Condition: {condition}",
            f"Inventory: {inventory if inventory else '(empty)'}",
        )
    )


def _render_grid_viewport(viewport: GridViewport, chars: np.ndarray) -> str:
    player = _find_in_viewport(chars, viewport, "@")
    player_text = (
        f"@ local row {player[0]}, col {player[1]}"
        if player is not None
        else "@ not visible"
    )
    return (
        "Grid view: terminal rows "
        f"{viewport.y0}-{viewport.y1 - 1}, cols {viewport.x0}-{viewport.x1 - 1} "
        f"of {viewport.full_height}x{viewport.full_width}; "
        f"{viewport.height}x{viewport.width} crop; {player_text}"
    )


def _find_in_viewport(
    chars: np.ndarray,
    viewport: GridViewport,
    target: str,
) -> tuple[int, int] | None:
    target_code = ord(target)
    cropped = chars[viewport.y0 : viewport.y1, viewport.x0 : viewport.x1]
    ys, xs = np.nonzero(cropped == target_code)
    if len(ys) == 0 or len(xs) == 0:
        return None
    return int(ys[0]), int(xs[0])


def _render_inventory(inv_strs: np.ndarray, inv_letters: np.ndarray) -> str:
    items: list[str] = []
    for i in range(inv_strs.shape[0]):
        letter = int(inv_letters[i])
        if letter == 0:
            break
        text = _decode_c_string(inv_strs[i])
        if text:
            items.append(f"{chr(letter)}: {text}")
    return "; ".join(items)


def _render_message(message: np.ndarray, tty_chars: np.ndarray) -> str:
    tty = _extract_tty_message(tty_chars)
    raw_message = _decode_c_string(message)
    if tty and raw_message and raw_message not in tty:
        return f"{raw_message}\n{tty}"
    return tty or raw_message


def _terminal_score_from_message(message: str) -> int | None:
    match = _TOP_TEN_SCORE_RE.search(message)
    if match is None:
        return None
    return int(match.group(1))


def _trim_tty_line(line: str) -> str:
    return line.strip(" \0\n\r\t\f\v")


def _extract_tty_message(tty_chars: np.ndarray) -> str:
    if tty_chars.shape[0] == 0:
        return ""
    lines = [
        "".join(chr(int(c)) if int(c) != 0 else " " for c in row)
        for row in tty_chars
    ]
    first = _trim_tty_line(lines[0]) if lines else ""
    second = _trim_tty_line(lines[1]) if len(lines) > 1 else ""
    if not first and not second:
        return ""

    indent_candidates = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    indent = min(indent_candidates) if indent_candidates else 0
    death_screen = "Points" in second or "list!" in second
    blank_rows = 0
    out: list[str] = []
    for line in lines:
        text = _trim_tty_line(line[indent:])
        if not text:
            blank_rows += 1
        else:
            blank_rows = 0
            out.append(text)
        if "--More--" in text or "(end)" in text or _looks_like_page_marker(text):
            return "\n".join(out)
        if death_screen and blank_rows > 1 and out:
            return "\n".join(out)
    return first


def _looks_like_page_marker(text: str) -> bool:
    start = text.rfind("(")
    end = text.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return False
    middle = text[start + 1:end]
    left, sep, right = middle.partition(" of ")
    return bool(sep and left.strip().isdigit() and right.strip().isdigit())
