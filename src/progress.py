import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class _DatasetState:
    label: str
    total: int
    done: int = 0
    ok: int = 0
    error: int = 0
    retries: int = 0
    last_task_id: str = "-"


class MultiDatasetProgress:
    def __init__(self, datasets: Sequence[Tuple[str, str, int]]):
        self._order: List[str] = []
        self._states: Dict[str, _DatasetState] = {}
        for dataset_name, display_label, total in datasets:
            self._order.append(dataset_name)
            self._states[dataset_name] = _DatasetState(
                label=display_label,
                total=max(total, 0),
            )

        self._is_tty = sys.stdout.isatty()
        self._rendered_lines = 0
        self._started = False
        self._start_time = time.perf_counter()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._order:
            self._render(force=True)

    def update(self, dataset_name: str, task_id: str, success: bool, retry_count: int) -> None:
        state = self._states.get(dataset_name)
        if state is None:
            return

        state.done += 1
        if success:
            state.ok += 1
        else:
            state.error += 1
        if retry_count > 0:
            state.retries += retry_count
        state.last_task_id = task_id or "-"

        if self._is_tty:
            self._render(force=False)
        else:
            if state.done == 1 or state.done == state.total or state.done % 10 == 0:
                self._print_single_line(dataset_name)

    def finish(self) -> None:
        if not self._started:
            return
        self._render(force=True)
        elapsed = time.perf_counter() - self._start_time
        print("Progress finished in {0:.1f}s".format(elapsed))

    def _render(self, force: bool) -> None:
        if not self._order:
            return
        lines = self._build_lines()
        block = "\n".join(lines) + "\n"

        if not self._is_tty:
            if force:
                print(block, end="")
            return

        if self._rendered_lines > 0:
            # Move cursor up to the first progress line then redraw full block.
            sys.stdout.write("\x1b[{0}F".format(self._rendered_lines))
        sys.stdout.write(block)
        sys.stdout.flush()
        self._rendered_lines = len(lines)

    def _build_lines(self) -> List[str]:
        width = shutil.get_terminal_size((160, 24)).columns
        lines: List[str] = []
        for idx, dataset_name in enumerate(self._order, start=1):
            state = self._states[dataset_name]
            total = state.total
            done = min(state.done, total) if total > 0 else state.done
            percent = (100.0 * done / total) if total > 0 else 100.0
            status = "DONE" if total == 0 or done >= total else "RUN"
            task_id = self._shorten(state.last_task_id, 40)
            line = (
                "{idx:02d}. {label:<42} | {status:<4} | {done:>4}/{total:<4} "
                "({percent:6.2f}%) | ok:{ok:<4} err:{err:<4} retry:{retry:<4} | {task_id}"
            ).format(
                idx=idx,
                label=self._shorten(state.label, 42),
                status=status,
                done=done,
                total=total,
                percent=percent,
                ok=state.ok,
                err=state.error,
                retry=state.retries,
                task_id=task_id,
            )
            lines.append(self._fit_line(line, width))
        return lines

    def _print_single_line(self, dataset_name: str) -> None:
        state = self._states[dataset_name]
        total = state.total
        done = min(state.done, total) if total > 0 else state.done
        percent = (100.0 * done / total) if total > 0 else 100.0
        print(
            "[{0}] {1}/{2} ({3:.2f}%) ok={4} err={5} retry={6} last={7}".format(
                state.label,
                done,
                total,
                percent,
                state.ok,
                state.error,
                state.retries,
                state.last_task_id,
            )
        )

    def _fit_line(self, line: str, width: int) -> str:
        if width <= 0:
            return line
        if len(line) <= width:
            return line
        return line[: max(0, width - 1)]

    def _shorten(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."


@dataclass
class _ValidationState:
    label: str
    total: int
    done: int = 0
    clean: int = 0
    hall: int = 0
    parse_error: int = 0
    last_task_id: str = "-"


class ValidationProgress:
    def __init__(self, files: Sequence[Tuple[str, str, int]]):
        self._order: List[str] = []
        self._states: Dict[str, _ValidationState] = {}
        for file_key, display_label, total in files:
            self._order.append(file_key)
            self._states[file_key] = _ValidationState(
                label=display_label,
                total=max(total, 0),
            )

        self._is_tty = sys.stdout.isatty()
        self._rendered_lines = 0
        self._started = False
        self._start_time = time.perf_counter()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._order:
            self._render(force=True)

    def update(
        self,
        file_key: str,
        task_id: str,
        has_hallucination: bool,
        parse_error: bool,
    ) -> None:
        state = self._states.get(file_key)
        if state is None:
            return

        state.done += 1
        if parse_error:
            state.parse_error += 1
        elif has_hallucination:
            state.hall += 1
        else:
            state.clean += 1
        state.last_task_id = task_id or "-"

        if self._is_tty:
            self._render(force=False)
        else:
            if state.done == 1 or state.done == state.total or state.done % 10 == 0:
                self._print_single_line(file_key)

    def finish(self) -> None:
        if not self._started:
            return
        self._render(force=True)
        elapsed = time.perf_counter() - self._start_time
        print("Validation progress finished in {0:.1f}s".format(elapsed))

    def _render(self, force: bool) -> None:
        if not self._order:
            return
        lines = self._build_lines()
        block = "\n".join(lines) + "\n"

        if not self._is_tty:
            if force:
                print(block, end="")
            return

        if self._rendered_lines > 0:
            sys.stdout.write("\x1b[{0}F".format(self._rendered_lines))
        sys.stdout.write(block)
        sys.stdout.flush()
        self._rendered_lines = len(lines)

    def _build_lines(self) -> List[str]:
        width = shutil.get_terminal_size((160, 24)).columns
        lines: List[str] = []
        for idx, file_key in enumerate(self._order, start=1):
            state = self._states[file_key]
            total = state.total
            done = min(state.done, total) if total > 0 else state.done
            percent = (100.0 * done / total) if total > 0 else 100.0
            status = "DONE" if total == 0 or done >= total else "RUN"
            task_id = self._shorten(state.last_task_id, 40)
            line = (
                "{idx:02d}. {label:<42} | {status:<4} | {done:>4}/{total:<4} "
                "({percent:6.2f}%) | hall:{hall:<4} clean:{clean:<4} parse_err:{parse_error:<4} | {task_id}"
            ).format(
                idx=idx,
                label=self._shorten(state.label, 42),
                status=status,
                done=done,
                total=total,
                percent=percent,
                hall=state.hall,
                clean=state.clean,
                parse_error=state.parse_error,
                task_id=task_id,
            )
            lines.append(self._fit_line(line, width))
        return lines

    def _print_single_line(self, file_key: str) -> None:
        state = self._states[file_key]
        total = state.total
        done = min(state.done, total) if total > 0 else state.done
        percent = (100.0 * done / total) if total > 0 else 100.0
        print(
            "[{0}] {1}/{2} ({3:.2f}%) hall={4} clean={5} parse_err={6} last={7}".format(
                state.label,
                done,
                total,
                percent,
                state.hall,
                state.clean,
                state.parse_error,
                state.last_task_id,
            )
        )

    def _fit_line(self, line: str, width: int) -> str:
        if width <= 0:
            return line
        if len(line) <= width:
            return line
        return line[: max(0, width - 1)]

    def _shorten(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."
