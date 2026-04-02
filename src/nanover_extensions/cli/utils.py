from pathlib import Path

from typing import Any, Iterable

import click


class _MultiPathParamType(click.ParamType):
    name = "tuple[Path]"

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> Any:
        if isinstance(value, Iterable) and not isinstance(value, str):
            if all(isinstance(el, Path) for el in value):
                return value

        try:
            return tuple(Path(el) for el in value)
        except ValueError:
            self.fail(f"Could not convert {', '.join(value)} to Path(s).", param, ctx)


MultiPath = _MultiPathParamType()


class OptionEatAll(click.Option):
    # Adapted from: https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, f"nargs, if set, must be -1 not {nargs}"
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser: Any | None = None

    def add_to_parser(self, parser: Any, ctx: click.Context) -> None:
        def parser_process(value: str, state: Any) -> None:
            # Method to hook to the parser.process
            done = False
            all_values = [value]
            if self.save_other_options:
                # Grab everything up to the next option
                while state.rargs and not done:
                    if self._eat_all_parser is None:
                        break

                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        all_values.append(state.rargs.pop(0))
            else:
                # Grab everything remaining
                all_values += state.rargs
                state.rargs[:] = []

            # call the actual process
            if self._previous_parser_process is not None:
                self._previous_parser_process(all_values, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval
