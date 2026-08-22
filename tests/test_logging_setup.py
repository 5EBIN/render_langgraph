import io
import logging

from render_langgraph.logging_setup import LOGGER_NAME, get_logger, setup_logging


def _reset():
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)


def test_default_hides_debug_shows_info_ok_warn_error():
    _reset()
    stream = io.StringIO()
    log = setup_logging(stream=stream)
    log.debug("debug line")
    log.info("info line")
    log.ok("ok line")
    log.warn("warn line")
    log.error("error line")

    out = stream.getvalue()
    assert "debug line" not in out
    assert "info line" in out
    assert "ok line" in out
    assert "warn line" in out
    assert "error line" in out


def test_verbose_shows_debug():
    _reset()
    stream = io.StringIO()
    log = setup_logging(verbose=True, stream=stream)
    log.debug("debug line")
    assert "debug line" in stream.getvalue()


def test_quiet_hides_info_and_ok_shows_warn_error():
    _reset()
    stream = io.StringIO()
    log = setup_logging(quiet=True, stream=stream)
    log.info("info line")
    log.ok("ok line")
    log.warn("warn line")
    log.error("error line")

    out = stream.getvalue()
    assert "info line" not in out
    assert "ok line" not in out
    assert "warn line" in out
    assert "error line" in out


def test_quiet_still_shows_errors_failures_are_always_loud():
    _reset()
    stream = io.StringIO()
    log = setup_logging(quiet=True, stream=stream)
    log.error("fatal detail here")
    assert "fatal detail here" in stream.getvalue()


def test_all_lines_prefixed_including_multiline_messages():
    _reset()
    stream = io.StringIO()
    log = setup_logging(stream=stream)
    log.warn("first line\nsecond line\nthird line")
    lines = [l for l in stream.getvalue().splitlines() if l.strip()]
    assert len(lines) == 3
    assert all(l.startswith("render-langgraph: ") for l in lines)


def test_color_enabled_on_tty_and_disabled_without(monkeypatch):
    _reset()

    class TTYStream(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    tty_stream = TTYStream()
    log = setup_logging(stream=tty_stream)
    log.ok("colored ok")
    assert "\033[32m" in tty_stream.getvalue()

    _reset()
    plain_stream = io.StringIO()  # isatty() is False on plain StringIO
    log = setup_logging(stream=plain_stream)
    log.ok("plain ok")
    assert "\033[" not in plain_stream.getvalue()


def test_no_color_env_var_disables_color_even_on_tty(monkeypatch):
    _reset()

    class TTYStream(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    stream = TTYStream()
    log = setup_logging(stream=stream)
    log.error("should not be colored")
    assert "\033[" not in stream.getvalue()


def test_get_logger_shares_configuration_across_modules():
    _reset()
    stream = io.StringIO()
    setup_logging(verbose=True, stream=stream)

    # A different module calling get_logger() (not setup_logging()) must
    # see the same handler/level -- this is the whole point of using
    # logging.getLogger(name) as a process-wide singleton.
    other = get_logger()
    other.debug("debug from another module")
    assert "debug from another module" in stream.getvalue()
