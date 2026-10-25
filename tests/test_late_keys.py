"""Keys typed while a fix is running are kept, not deleted over.

A fix takes a few hundred milliseconds and the user keeps typing. Those keys
used to reach the application in the middle of the count-based rewrite, which
then deleted them plus part of the word and typed the correction in the wrong
place: ``ghbdtn vbh `` became ``ghbпривет `` (reproduced on GNOME 51).

Now the keyboards are grabbed for the fix and what was typed meanwhile is
replayed on the virtual keyboard afterwards; letters that slipped through
before the grab are deleted with the word and retyped after it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

evdev = pytest.importorskip("evdev")

from klyrshift import daemon as daemon_module  # noqa: E402
from klyrshift.config import Config  # noqa: E402
from klyrshift.daemon import KlyrshiftDaemon  # noqa: E402
from klyrshift.injector import TextInjector  # noqa: E402
from klyrshift.switcher import LayoutSwitcher  # noqa: E402

EV_KEY = int(evdev.ecodes.EV_KEY)


@dataclass
class FakeEvent:
    type: int
    code: int
    value: int
    stamp: float = 0.0

    def timestamp(self) -> float:
        return self.stamp


def press(name: str, stamp: float = 0.0) -> list[FakeEvent]:
    code = int(getattr(evdev.ecodes, name))
    return [FakeEvent(EV_KEY, code, 1, stamp), FakeEvent(EV_KEY, code, 0, stamp)]


@dataclass
class FakeKeyboard:
    """A grabbable keyboard. ``during_grab`` is delivered only while grabbed.

    A ``None`` in ``pending`` means "nothing more yet": one ``read_one`` call
    returns ``None`` and the next one continues with what follows.
    """

    log: list[str]
    pending: list[FakeEvent | None] = field(default_factory=list)
    during_grab: list[FakeEvent | None] = field(default_factory=list)
    grabbed: bool = False

    def grab(self) -> None:
        self.grabbed = True
        self.log.append("grab")
        self.pending.extend(self.during_grab)
        self.during_grab = []

    def ungrab(self) -> None:
        self.grabbed = False
        self.log.append("ungrab")

    def read_one(self) -> FakeEvent | None:
        return self.pending.pop(0) if self.pending else None


class Switcher(LayoutSwitcher):
    def __init__(self, log: list[str]) -> None:
        super().__init__(["us", "ru"], "none")
        self.log = log
        self.layout = "us"

    def get_current_layout(self, *, force: bool = False) -> str:
        return self.layout

    def start_switch(self, layout: str) -> Callable[[], bool]:
        self.log.append(f"start {layout}")

        def finish() -> bool:
            self.layout = layout
            return True

        return finish


class Injector(TextInjector):
    def __init__(self, log: list[str]) -> None:
        super().__init__(backend="none")
        self._backend = "uinput"
        self.log = log
        self.forwarded: list[FakeEvent] = []

    def replace_text(
        self,
        backspace_count: int,
        new: str,
        layout: str,
        before_typing: Callable[[], bool] | None = None,
    ) -> bool:
        if before_typing is not None:
            before_typing()
        self.log.append(f"replace {backspace_count} {new!r}")
        return True

    def forward_events(self, events: list[object]) -> bool:
        self.forwarded.extend(events)
        self.log.append(f"forward {len(events)}")
        return True


class Selector:
    def __init__(self, keyboard: FakeKeyboard) -> None:
        self.keyboard = keyboard

    def get_map(self) -> dict[int, SimpleNamespace]:
        return {3: SimpleNamespace(data=self.keyboard)}


@pytest.fixture
def log() -> list[str]:
    return []


@pytest.fixture
def keyboard(log: list[str]) -> FakeKeyboard:
    return FakeKeyboard(log)


@pytest.fixture
def daemon(
    log: list[str], keyboard: FakeKeyboard, monkeypatch: pytest.MonkeyPatch
) -> KlyrshiftDaemon:
    monkeypatch.setattr(daemon_module.time, "sleep", lambda _seconds: None)
    d = KlyrshiftDaemon(Config(stop_words=[]), switcher=Switcher(log), injector=Injector(log))
    d._selector = Selector(keyboard)  # type: ignore[assignment]
    return d


def _injector(d: KlyrshiftDaemon) -> Injector:
    assert isinstance(d.injector, Injector)
    return d.injector


def test_fix_runs_inside_a_grab(daemon: KlyrshiftDaemon, log: list[str]) -> None:
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    assert log == ["grab", "start ru", "replace 7 'привет '", "ungrab"]


def test_keys_typed_during_the_grab_are_replayed_and_queued(
    daemon: KlyrshiftDaemon, keyboard: FakeKeyboard, log: list[str]
) -> None:
    later = time.time() + 3600  # stamped after the grab started
    typed = press("KEY_V", later) + press("KEY_B", later) + press("KEY_SPACE", later)
    keyboard.during_grab = list(typed)
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    # The fix itself is untouched by the keys typed meanwhile ...
    assert "replace 7 'привет '" in log
    # ... which are replayed after it, before the keyboard is released.
    assert log.index("replace 7 'привет '") < log.index("forward 6") < log.index("ungrab")
    assert _injector(daemon).forwarded == typed
    # And handed to the loop as the next word.
    assert list(daemon._event_queue) == typed


def test_letters_before_the_grab_are_deleted_and_retyped(
    daemon: KlyrshiftDaemon, log: list[str]
) -> None:
    early = press("KEY_V") + press("KEY_B")
    daemon._event_queue.extend(early)
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    # Two more Backspaces for "vb", typed back as "ми" in the new layout.
    assert "replace 9 'привет ми'" in log
    # They already reached the application: queued, not replayed.
    assert _injector(daemon).forwarded == []
    assert list(daemon._event_queue) == early


def test_shift_before_the_grab_is_followed(daemon: KlyrshiftDaemon, log: list[str]) -> None:
    shift = int(evdev.ecodes.KEY_LEFTSHIFT)
    daemon._event_queue.extend(
        [FakeEvent(EV_KEY, shift, 1), *press("KEY_V"), FakeEvent(EV_KEY, shift, 0), *press("KEY_B")]
    )
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    assert "replace 9 'привет Ми'" in log


@pytest.mark.parametrize(
    "key", ["KEY_BACKSPACE", "KEY_LEFT", "KEY_SPACE", "KEY_LEFTCTRL", "KEY_ENTER"]
)
def test_anything_but_letters_before_the_grab_skips_the_fix(
    daemon: KlyrshiftDaemon, log: list[str], key: str
) -> None:
    early = press("KEY_V") + press(key)
    daemon._event_queue.extend(early)
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is False
    assert not any(entry.startswith("replace") for entry in log)
    assert log[-1] == "ungrab"
    assert list(daemon._event_queue) == early


def test_grab_waits_for_held_keys_and_is_held_until_they_are_released(
    daemon: KlyrshiftDaemon, keyboard: FakeKeyboard, log: list[str]
) -> None:
    space = int(evdev.ecodes.KEY_SPACE)
    v = int(evdev.ecodes.KEY_V)
    later = time.time() + 3600
    pending_at_grab: list[list[FakeEvent | None]] = []
    original_grab = keyboard.grab

    def grab() -> None:
        # The Space release must have been read before the grab.
        pending_at_grab.append(list(keyboard.pending))
        original_grab()

    keyboard.grab = grab  # type: ignore[method-assign]
    daemon._keys_down.add(space)  # the boundary Space is still down
    keyboard.pending = [None, FakeEvent(EV_KEY, space, 0)]
    # V is pressed during the fix and released only after a pause.
    keyboard.during_grab = [
        FakeEvent(EV_KEY, v, 1, later),
        None,
        None,
        FakeEvent(EV_KEY, v, 0, later),
    ]
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    assert pending_at_grab == [[]]
    # Grabbed until V was released: its press and release are both replayed.
    assert _injector(daemon).forwarded == [
        FakeEvent(EV_KEY, v, 1, later),
        FakeEvent(EV_KEY, v, 0, later),
    ]
    assert log[-1] == "ungrab"
    # The Space release and V go back to the loop, in order.
    assert [(e.code, e.value) for e in daemon._event_queue] == [(space, 0), (v, 1), (v, 0)]


def test_no_grab_while_a_key_stays_held(
    daemon: KlyrshiftDaemon, keyboard: FakeKeyboard, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_module, "GRAB_KEYS_UP_TIMEOUT", 0.0)
    daemon._keys_down.add(int(evdev.ecodes.KEY_SPACE))
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    assert log == ["start ru", "replace 7 'привет '"]


def test_key_state_is_tracked_from_handled_events(daemon: KlyrshiftDaemon) -> None:
    """``active_keys`` (EVIOCGKEY) flushes queued events, so it is never used."""
    v = int(evdev.ecodes.KEY_V)
    daemon._handle_event(FakeEvent(EV_KEY, v, 1))
    assert daemon._keys_down == {v}
    daemon._handle_event(FakeEvent(EV_KEY, v, 2))
    assert daemon._keys_down == {v}
    daemon._handle_event(FakeEvent(EV_KEY, v, 0))
    assert daemon._keys_down == set()


def test_events_grabbed_just_before_the_ungrab_are_replayed(
    daemon: KlyrshiftDaemon, keyboard: FakeKeyboard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read after the ungrab but stamped before it: the app never saw them."""
    original_ungrab = keyboard.ungrab
    now = time.time()
    grabbed_late = press("KEY_H", stamp=now - 1)
    not_grabbed = press("KEY_J", stamp=now + 3600)

    def ungrab() -> None:
        original_ungrab()
        keyboard.pending.extend(grabbed_late + not_grabbed)

    keyboard.ungrab = ungrab  # type: ignore[method-assign]
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    assert _injector(daemon).forwarded == grabbed_late
    assert list(daemon._event_queue) == grabbed_late + not_grabbed


def test_no_grab_without_keep_late_keys(daemon: KlyrshiftDaemon, log: list[str]) -> None:
    assert daemon._switch_and_replace(7, "привет ", "ru") is True
    assert "grab" not in log


def test_no_grab_for_devices_that_cannot_be_grabbed(
    daemon: KlyrshiftDaemon, log: list[str]
) -> None:
    daemon._selector = Selector(SimpleNamespace(read_one=lambda: None))  # type: ignore[assignment]
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    assert log == ["start ru", "replace 7 'привет '"]


def test_a_failing_grab_falls_back_and_releases(
    daemon: KlyrshiftDaemon, keyboard: FakeKeyboard, log: list[str]
) -> None:
    def grab() -> None:
        raise OSError("busy")

    keyboard.grab = grab  # type: ignore[method-assign]
    assert daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True) is True
    assert log == ["start ru", "replace 7 'привет '"]


def test_the_keyboard_is_released_when_the_fix_raises(
    daemon: KlyrshiftDaemon, keyboard: FakeKeyboard, log: list[str]
) -> None:
    def boom(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("injector exploded")

    _injector(daemon).replace_text = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        daemon._switch_and_replace(7, "привет ", "ru", keep_late_keys=True)
    assert log[-1] == "ungrab"
    assert keyboard.grabbed is False


def test_forward_events_writes_key_events_on_the_shared_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    written: list[tuple[int, int]] = []

    class UInput:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def write(self, _type: int, code: int, value: int) -> None:
            written.append((code, value))

        def syn(self) -> None:
            pass

    fake = types.ModuleType("evdev")
    fake.ecodes = evdev.ecodes
    fake.UInput = UInput
    monkeypatch.setitem(sys.modules, "evdev", fake)
    monkeypatch.setattr(TextInjector, "_uinput_available", lambda self: True)
    injector = TextInjector(backend="uinput")
    v = int(evdev.ecodes.KEY_V)
    events = [FakeEvent(EV_KEY, v, 1), FakeEvent(0, 0, 0), FakeEvent(EV_KEY, v, 0)]
    assert injector.forward_events(events) is True
    assert written == [(v, 1), (v, 0)]
    monkeypatch.setattr("klyrshift.injector._which", lambda name: "/usr/bin/" + name)
    assert TextInjector(backend="wtype").forward_events(events) is False
