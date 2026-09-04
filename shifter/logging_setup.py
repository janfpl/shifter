"""Terminal logging and crash diagnostics for the Shifter application.

By default the application configures no logging handler, so every
``logging.getLogger(...).<level>(...)`` call across the package is silently
dropped, and an exception raised inside the Qt event loop (for example in a
signal/slot callback) can abort the process with no Python traceback — the
classic "napari just closed and I don't know why".

This module wires up a stderr log handler and installs hooks so that uncaught
exceptions — including those raised inside Qt slots and Qt's own fatal
messages — are printed with a full traceback instead of vanishing.

Set the ``SHIFTER_LOG_LEVEL`` environment variable (e.g. ``DEBUG``) to raise
verbosity.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys

logger = logging.getLogger("shifter")

_LEVEL_ENV = "SHIFTER_LOG_LEVEL"
_QT_LOG_ENV = "SHIFTER_QT_LOG"  # set to "0" to disable the Qt message handler
_configured = False


def configure_logging(level: int | str | None = None) -> None:
    """Attach a stderr handler to the root logger (idempotent).

    Respects an existing logging configuration: if the caller (or napari) has
    already added root handlers, only the level is adjusted so we don't emit
    duplicate lines. Also enables :mod:`faulthandler`, so that a *native* crash
    (segfault or C++ ``abort`` from Qt/OpenGL, which Python's exception handling
    cannot catch) dumps the crashing thread's stack to stderr instead of the
    process simply vanishing.
    """
    global _configured
    if level is None:
        level = os.environ.get(_LEVEL_ENV, "INFO")

    root = logging.getLogger()
    if not _configured and not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(level)

    if not faulthandler.is_enabled():
        try:
            faulthandler.enable()
        except Exception:  # pragma: no cover - depends on stderr being real
            pass

    _configured = True


def install_excepthook() -> None:
    """Route uncaught exceptions through logging so they always reach stderr.

    With PyQt an exception that escapes a slot otherwise calls the default
    ``sys.excepthook``, which prints and then aborts the process; replacing it
    ensures the traceback is logged (and, for slot exceptions, that the app is
    not torn down before the message is seen).
    """
    previous = sys.excepthook

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook


def install_qt_message_handler() -> None:
    """Forward Qt's own debug/warning/fatal messages into the Python log.

    No-op if Qt bindings are not importable yet, or if ``SHIFTER_QT_LOG=0`` is
    set (a kill-switch for isolating whether this hook itself is implicated in a
    startup crash). The callback is defensive: it never lets a logging failure
    propagate back into the Qt C++ layer, which would turn a warning into an
    abort.
    """
    if os.environ.get(_QT_LOG_ENV) == "0":
        return
    try:
        from qtpy.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return

    qt_logger = logging.getLogger("shifter.qt")
    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _handler(mode, context, message):  # noqa: ANN001 - Qt callback signature
        try:
            qt_logger.log(level_map.get(mode, logging.INFO), "%s", message)
        except Exception:
            pass  # never propagate a logging error into Qt's C++ layer

    qInstallMessageHandler(_handler)


def install_diagnostics() -> None:
    """Install the exception hook and the Qt message handler together."""
    install_excepthook()
    install_qt_message_handler()
