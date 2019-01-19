import json
import collections
import contextlib
from datetime import datetime, timedelta
import itertools


def merge_dicts(*dicts):
    result = {}
    for _dict in dicts:
        result.update(_dict)

    return result


def pretty_duration(duration, present_only=False):
    total_seconds = duration.total_seconds()
    units = [
        ('d', round(total_seconds // (60 * 60 * 24))),
        ('h', round(total_seconds // (60 * 60)) % 24),
        ('m', round(total_seconds // 60) % 60),
        ('s', round(total_seconds % 60, 2)),
    ]

    if present_only:
        while len(units) > 1 and units[-1][1] == 0:
            units = units[:-1]

    return "".join("{}{}".format(value, unit) for unit, value in units)


class CannotPlayError(Exception):
    pass


class CannotMoveError(CannotPlayError):
    pass


class CannotBuildError(CannotPlayError):
    pass


class Board:
    size = NotImplemented
    positions = NotImplemented
    transformations = NotImplemented
    reverse_transformations = NotImplemented
    max_level = NotImplemented
    win_level = NotImplemented

    PLAYER_A = 'a'
    PLAYER_B = 'b'

    @classmethod
    def for_size(cls, size):
        _size = size

        class MapForSize(cls):
            size = _size
            positions = [
                (row, column)
                for column in range(size)
                for row in range(size)
            ]
        MapForSize.__name__ = "{}{}".format(cls.__name__, size)

        MapForSize.plays = MapForSize.get_all_plays()
        MapForSize.transformations, MapForSize.reverse_transformations = \
            MapForSize.get_all_transformations()

        return MapForSize

    @classmethod
    def for_max_level(cls, max_level):
        _max_level = max_level

        class MapForMaxLevel(cls):
            max_level = _max_level
            win_level = _max_level - 1
            REPR_LEVEL_MAP = merge_dicts({0: ' '}, {
                level: str(level)
                for level in range(1, _max_level + 1)
            })
            HASH_LEVEL_MAP = merge_dicts({0: ' '}, {
                level: str(level)
                for level in range(1, _max_level + 1)
            })
            PARSE_LEVEL_MAP = merge_dicts({' ': 0}, {
                str(level): level
                for level in range(1, _max_level + 1)
            })
        MapForMaxLevel.__name__ = "{}{}".format(cls.__name__, max_level)

        return MapForMaxLevel

    @classmethod
    def get_initial_boards(cls):
        initial_positions_a = list(itertools.permutations(cls.positions, 4))
        initial_positions_b = [
            (a1, a2, b1, b2)
            for b1, b2, a1, a2 in initial_positions_a
        ]
        initial_positions = initial_positions_a + initial_positions_b
        return tuple({
            cls.from_initial_position(*initial_position)
            for initial_position in initial_positions
        })

    @classmethod
    def from_initial_position(cls, a1, a2, b1, b2):
        return cls({
            a1: (cls.PLAYER_A, 0),
            a2: (cls.PLAYER_A, 0),
            b1: (cls.PLAYER_B, 0),
            b2: (cls.PLAYER_B ,0),
        })

    offsets = [
        (row, column)
        for row in range(-1, 2)
        for column in range(-1, 2)
        if (row, column) != (0, 0)
    ]
    plays = NotImplemented

    REPR_PLAYER_MAP = {None: ' ', PLAYER_A: 'a', PLAYER_B: 'b'}
    REPR_LEVEL_MAP = NotImplemented

    HASH_PLAYER_MAP = {None: ' ', PLAYER_A: 'a', PLAYER_B: 'b'}
    HASH_LEVEL_MAP = NotImplemented

    PARSE_PLAYER_MAP = {' ': None, 'a': PLAYER_A, 'b': PLAYER_B}
    PARSE_LEVEL_MAP = NotImplemented

    SWITCH_PLAYER_MAP = {None: None, PLAYER_A: PLAYER_B, PLAYER_B: PLAYER_A}

    @classmethod
    def get_opposite_player(cls, player):
        return cls.SWITCH_PLAYER_MAP[player]

    @classmethod
    def from_board_hash(cls, board_hash):
        values = [
            (cls.PARSE_PLAYER_MAP[player_str], cls.PARSE_LEVEL_MAP[level_str])
            for player_str, level_str in (
                board_hash[start:start + 2]
                for start in range(0, len(board_hash), 2)
            )
        ]
        new_values = dict(zip(cls.positions, values))
        board = cls(new_values)
        board._board_hash = board_hash

        return board

    @classmethod
    def from_equivalent_hash(cls, equivalent_hash):
        board = cls.from_board_hash(equivalent_hash)
        board._equivalent_hash = equivalent_hash

        return board

    __slots__ = [
        '_board_hash',
        '_equivalent_hash',
        '_has_player_b_won',
        '_move_count',
        'board',
    ]

    def __init__(self, values=None):
        self._board_hash = None
        self._equivalent_hash = None
        self._has_player_b_won = None
        self._move_count = None

        if values is None:
            values = {}
        self.board = tuple(
            tuple(
                values.get((row, column), (None, 0))
                for column in range(self.size)
            )
            for row in range(self.size)
        )

    def __eq__(self, other):
        return self.board == other.board

    def __repr__(self):
        return "+{0}+\n{1}\n+{0}+".format(
            " ".join(['--'] * self.size),
            "\n".join(
                "|{}|".format(" ".join(
                    "{}{}".format(
                        self.REPR_PLAYER_MAP[player],
                        self.REPR_LEVEL_MAP[level],
                    )
                    for player, level in row
                ))
                for row in self.board
            ),
        )

    @property
    def move_count(self):
        if self._move_count is None:
            self._move_count = sum(
                level
                for row in self.board
                for _, level in row
            )
        return self._move_count


    @property
    def board_hash(self):
        if not self._board_hash:
            self._board_hash =  self.get_board_hash()

        return self._board_hash

    def get_board_hash(self):
        return "".join(
            "{}{}".format(
                self.HASH_PLAYER_MAP[player],
                self.HASH_LEVEL_MAP[level],
            )
            for player, level in (
                self[position]
                for position in self.positions
            )
        )

    def get_board_hash_with_reverse_transformation(
            self, reverse_transformation):
        return "".join(
            "{}{}".format(
                self.HASH_PLAYER_MAP[player],
                self.HASH_LEVEL_MAP[level],
            )
            for player, level in (
                self[reverse_transformation[position]]
                for position in self.positions
            )
        )

    @property
    def equivalent_hash(self):
        if not self._equivalent_hash:
            self._equivalent_hash = self.get_equivalent_hash()

        return self._equivalent_hash

    def get_equivalent_hash(self):
        return min(
            self.get_board_hash_with_reverse_transformation(
                reverse_transformation)
            for reverse_transformation in self.reverse_transformations
        )

    def __hash__(self):
        return hash(self.equivalent_hash)

    def __getitem__(self, position):
        row, column = position
        return self.board[row][column]

    def get(self, position, default=None):
        return self[position]

    def __setitem__(self, position, value):
        row, column = position
        self.board[row][column] = value

    def all_equivalents(self):
        return [
            self.transform_with_transformation(transformation)
            for transformation in self.transformations
        ]

    def rotate_cw(self):
        return self.transform(self.rotate_cw_position_transformation)

    @classmethod
    def rotate_cw_position_transformation(cls, row, column):
        return (column, cls.size - 1 - row)

    def rotate_ccw(self):
        return self.trasnform(self.rotate_ccw_position_transformation)

    @classmethod
    def rotate_ccw_position_transformation(cls, row, column):
        return (cls.size - 1 - column, row)

    def mirror_horizontal(self):
        return self.transform(self.mirror_horizontal_position_transformation)

    @classmethod
    def mirror_horizontal_position_transformation(cls, row, column):
        return (cls.size - 1 - row, column)

    def mirror_vertical(self):
        return self.transform(self.mirror_vertical_position_transformation)

    @classmethod
    def mirror_vertical_position_transformation(cls, row, column):
        return (row, cls.size - 1 - column)

    def transform(self, position_transformation):
        return type(self)({
            position_transformation(*input_position): self[input_position]
            for input_position in self.positions
        })

    def transform_with_transformation(self, transformation):
        return type(self)({
            output_position: self[input_position]
            for output_position, input_position in transformation.items()
        })

    @classmethod
    def apply_transformation(cls, source, position_transformation):
        return {
            position_transformation(*input_position): source[input_position]
            for input_position in cls.positions
        }

    @classmethod
    def get_all_transformations(cls):
        rotations = []

        original = {
            position: position
            for position in cls.positions
        }

        rotation = original
        rotations.append(rotation)
        for _ in range(3):
            rotation = cls.apply_transformation(
                rotation, cls.rotate_cw_position_transformation)
            rotations.append(rotation)

        transformations = rotations + [
            cls.apply_transformation(
                rotation, cls.mirror_horizontal_position_transformation)
            for rotation in rotations
        ]

        reverse_transformations = [
            {
                input_position: output_position
                for output_position, input_position in transformation.items()
            }
            for transformation in transformations
        ]

        return transformations, reverse_transformations

    def mutate(self, position_mutation):
        return type(self)({
            position: position_mutation(*(list(self[position]) + list(position)))
            for position in self.positions
        })

    def combine_position_mutations(self, position_mutations):
        def combined_position_mutations(player, level, row, column):
            for position_mutation in position_mutations:
                player, level = position_mutation(player, level, row, column)

            return (player, level)

        return combined_position_mutations

    def play(self, start_position, end_position, build_position):
        play = (start_position, end_position, build_position)
        if play not in self.plays:
            raise CannotPlayError("Play is not a valid play")
        if self.has_player_b_won():
            raise CannotPlayError("Player B has already won")

        return self.mutate(self.combine_position_mutations([
            self.get_move_position_mutation(start_position, end_position),
            self.get_build_position_mutation(end_position, build_position),
            self.switch_players_position_mutation,
        ]))

    def has_player_b_won(self):
        if self._has_player_b_won is None:
            winning_value = (self.PLAYER_B, self.win_level)
            self._has_player_b_won = any(
                winning_value in row
                for row in self.board
            )
        return self._has_player_b_won

    def move(self, start_position, end_position):
        if start_position == end_position:
            raise CannotMoveError(
                "Cannot move from {} to {} since they are the same".format(
                    start_position, end_position))

        start_row, start_column = start_position
        end_row, end_column = end_position
        if abs(start_row - end_row) > 1 or abs(start_column - end_column) > 1:
            raise CannotMoveError(
                "Cannot move from {} to {} since they are too far away".format(
                    start_position, end_position))

        start_player, start_level = self[start_position]
        if start_player != self.PLAYER_A:
            raise CannotMoveError(
                "Cannot move from {} since it's not occupied by player A"
                .format(start_position))

        end_player, end_level = self[end_position]
        if end_player is not None:
            raise CannotMoveError(
                "Cannot move to {} since it's occupied by player '{}'".format(
                    end_position, end_player))
        if end_level > (start_level + 1):
            raise CannotMoveError(
                "Cannot move from {} to {} since it's more than one level " \
                "up: {} -> {}".format(
                    start_position, end_position, start_level, end_level))

        return self.mutate(self.get_move_position_mutation(
            start_position, end_position))

    def get_move_position_mutation(self, start_position, end_position):
        def position_mutation(player, level, row, column):
            if (row, column) == start_position:
                return (None, level)
            if (row, column) == end_position:
                return (self.PLAYER_A, level)
            return (player, level)

        return position_mutation

    def build(self, end_position, build_position):
        if end_position == build_position:
            raise CannotBuildError(
                "Cannot build from {} to {} since they are the same".format(
                    end_position, build_position))

        end_row, end_column = end_position
        build_row, build_column = build_position
        if abs(end_row - build_row) > 1 or abs(end_column - build_column) > 1:
            raise CannotBuildError(
                "Cannot build from {} to {} since they are too far away".format(
                    end_position, build_position))

        end_player, end_level = self[end_position]
        if end_player != self.PLAYER_A:
            raise CannotBuildError(
                "Cannot build from {} since it's not occupied by player A"
                .format(end_position))

        build_player, build_level = self[build_position]
        if build_player is not None:
            raise CannotBuildError(
                "Cannot build to {} since it's occupied by player '{}'".format(
                    build_position, build_player))
        if build_level == self.max_level:
            raise CannotBuildError(
                "Cannot build to {} since it's already at the maximum level"
                .format(build_position))

        return self.mutate(self.get_build_position_mutation(
            end_position, build_position))

    def get_build_position_mutation(self, end_position, build_position):
        def position_mutation(player, level, row, column):
            if (row, column) == build_position:
                return (player, level + 1)
            return (player, level)

        return position_mutation

    def switch_players(self):
        return self.mutate(self.switch_players_position_mutation)

    def switch_players_position_mutation(self, player, level, row, column):
        return (self.SWITCH_PLAYER_MAP[player], level)

    @classmethod
    def get_all_plays(cls):
        return {
            play
            for play in (
                (start_position, end_position,
                 cls.add_positions(end_position, build_offset))
                for start_position, end_position in (
                    (start_position,
                     cls.add_positions(start_position, move_offset))
                    for start_position in cls.positions
                    for move_offset in cls.offsets
                )
                for build_offset in cls.offsets
            )
            if all(play)
        }

    def get_possible_plays(self):
        return [
            (start_position, end_position, build_position)
            for start_position, end_position, build_position in self.plays
            if self[start_position][0] == self.PLAYER_A
            and self[end_position][0] is None
            and self[end_position][1] <= (self[start_position][1] + 1)
            and (
                build_position == start_position
                or self[build_position][0] is None
            )
            and self[build_position][1] < self.max_level
        ]

    @classmethod
    def add_positions(cls, lhs, rhs):
        if not lhs or not rhs:
            return None

        l_row, l_column = lhs
        r_row, r_column = rhs
        row = l_row + r_row
        column = l_column + r_column

        if not (0 <= row < cls.size):
            return None
        if not (0 <= column < cls.size):
            return None

        return row, column

    def get_next_boards(self):
        next_boards = set()

        for play in self.get_possible_plays():
            try:
                next_board = self.play(*play)
            except CannotPlayError:
                pass
            else:
                next_boards.add(next_board)
        return list(next_boards)

Board64 = Board.for_size(6).for_max_level(4)

Board31 = Board.for_size(3).for_max_level(1)
Board32 = Board.for_size(3).for_max_level(2)
Board33 = Board.for_size(3).for_max_level(3)
Board34 = Board.for_size(3).for_max_level(4)


class BaseQueue:
    AUTO_SAVE_INTERVAL = timedelta(hours=4)

    __slots__ = [
        'initial_board',
        'board_type',
        'saving',
        'saving_to',
        'previous_auto_save_time',
        'iteration',
        'queue_count',
        'seen_count',
        'result_count',
        'previous_iteration',
        'previous_queue_count',
        'previous_seen_count',
        'previous_result_count',
        'last_board',
        'last_time',
    ]

    def __init__(self, initial_board, saving=False, saving_to=None):
        self.initial_board = initial_board
        self.board_type = type(self.initial_board)

        self.saving = False
        if saving_to is None:
            saving_to = './resume-{0}x{0}-max{1}.json'.format(
                self.board_type.size, self.board_type.max_level)
        self.saving_to = saving_to
        self.previous_auto_save_time = None

        self.iteration = None
        self.queue_count = None
        self.seen_count = None
        self.result_count = None
        self.previous_iteration = None
        self.previous_queue_count = None
        self.previous_seen_count = None
        self.previous_result_count = None
        self.last_board = None
        self.last_time = None

    def save(self, filename=None):
        if filename is None:
            filename = self.saving_to
        print("Saving to '{}'...".format(filename))
        start_time = datetime.now()
        with open(filename, 'w') as f:
            json.dump(self.get_state(), f)
        print("Saved in {}".format(
            pretty_duration(datetime.now() - start_time, present_only=True)))

    @classmethod
    def load(cls, filename=None, resuming=False):
        if filename is None:
            filename = self.saving_to
        print("Loading from '{}'...".format(filename))
        start_time = datetime.now()
        with open(filename, 'r') as f:
            state = json.load(f)

        queue = cls.from_state(state, filename=filename, resuming=resuming)
        print("Loaded in {}".format(
            pretty_duration(datetime.now() - start_time, present_only=True)))

        return queue

    @classmethod
    def from_state(cls, state, filename=None, resuming=False):
        board_type = Board\
            .for_size(state['board_type']['size'])\
            .for_max_level(state['board']['max_level'])
        initial_board = board_type.from_board_hash(state['initial_board'])
        queue = cls(initial_board)

        queue.apply_state(state, filename=filename, resuming=resuming)

        return queue

    def get_state(self):
        return {
            'saving': self.saving,
            'initial_board': self.initial_board.board_hash,
            'board_type': {
                'size': self.board_type.size,
                'max_level': self.board_type.max_level,
            },
            'iteration': self.iteration,
            'queue_count': self.queue_count,
            'seen_count': self.seen_count,
            'result_count': self.result_count,
            'previous_iteration': self.previous_iteration,
            'previous_queue_count': self.previous_queue_count,
            'previous_seen_count': self.previous_seen_count,
            'previous_result_count': self.previous_result_count,
            'last_board': self.last_board.board_hash,
        }

    def apply_state(self, state, filename=None, resuming=False):
        if resuming:
            queue.saving = state['saving']
            queue.saving_to = filename
        queue.iteration = state['iteration']
        queue.queue_count = state['queue_count']
        queue.seen_count = state['seen_count']
        queue.result_count = state['result_count']
        queue.previous_iteration = state['previous_iteration']
        queue.previous_queue_count = state['previous_queue_count']
        queue.previous_seen_count = state['previous_seen_count']
        queue.previous_result_count = state['previous_result_count']
        queue.last_board = board_type.from_board_hash(state['last_board'])

    def export_solution(self, filename=None):
        if filename is None:
            filename = './solved-{0}x{0}-max{1}.json'.format(
                self.initial_board.size, self.initial_board.max_level)
        with open(filename, 'w') as f:
            json.dump({
                'start': self.initial_board.equivalent_hash,
                'forward': self.forward,
                'result': merge_dicts({
                    h: 'a'
                    for h in self.result_a
                }, {
                    h: 'b'
                    for h in self.result_b
                }),
            }, f, indent=4)

    def start(self):
        self.clear()
        self.push_board(self.initial_board, None)
        return self.resume()

    def resume(self):
        return self.run_all()

    def clear(self):
        self.clear_queue()
        self.clear_stats()

    def clear_queue(self):
        raise NotImplementedError()

    def clear_stats(self):
        self.iteration = 0
        self.queue_count = 0
        self.seen_count = 0
        self.result_count = 0
        self.previous_iteration = 0
        self.previous_queue_count = 0
        self.previous_seen_count = 0
        self.previous_result_count = 0
        self.last_board = None
        self.last_time = None
        self.previous_auto_save_time = None

    def run_all(self):
        self.print_stats(force=True)
        while not self.has_initial_board_resulted() and self.run_one():
            self.print_stats()
            self.auto_save()
        self.print_stats(force=True)
        return self.get_board_result(self.initial_board)

    def print_stats(self, force=False):
        if not self.should_print_stats(force=force):
            return

        delta_iteration = self.iteration - self.previous_iteration
        delta_queue_count = self.queue_count - self.previous_queue_count
        delta_seen_count = self.seen_count - self.previous_seen_count
        delta_result_count = self.result_count - self.previous_result_count

        this_time = datetime.now()
        if self.last_time is not None:
            duration = this_time - self.last_time
            duration_total_seconds = duration.total_seconds()
            print("{} After {}:".format("-" * 10, pretty_duration(duration)))
        else:
            duration = None
            duration_total_seconds = 0
            print("-" * 35)
        self.last_time = this_time
        print("{:10}: {:10} ({:5}, {:5}/s)".format("Iteration", self.iteration, delta_iteration, round(delta_iteration / duration_total_seconds) if duration_total_seconds else 0))
        print("{:10}: {:10} ({:5})".format("Queue", self.queue_count, delta_queue_count))
        print("{:10}: {:10} ({:5}, {}%)".format("Seen", self.seen_count, delta_seen_count, round(100 * (self.seen_count - self.queue_count) / self.seen_count) if self.seen_count else 'N/A'))
        print("{:10}: {:10} ({:5}, {}%)".format("Resulted", self.result_count, delta_result_count, round(100 * self.result_count / self.seen_count) if self.seen_count else 'N/A'))
        print_board = self.last_board or self.initial_board
        print('{} Moves ({}, max {})'.format(print_board.move_count, "{0}x{0}".format(print_board.size), print_board.max_level))
        print(print_board)
        print("-" * 35)

        self.previous_iteration = self.iteration
        self.previous_queue_count = self.queue_count
        self.previous_seen_count = self.seen_count
        self.previous_result_count = self.result_count

    def should_print_stats(self, force):
        if self.iteration % 2500 != 0:
            return False

        delta_queue_count = self.queue_count - self.previous_queue_count
        delta_seen_count = self.seen_count - self.previous_seen_count
        delta_result_count = self.result_count - self.previous_result_count

        return force or (
            abs(delta_seen_count) >= 10000
            or abs(delta_queue_count) >= 10000
            or abs(delta_result_count) >= 10000
        )

    def auto_save(self):
        if not self.should_auto_save():
            return

        self.save(self.saving_to)
        self.previous_auto_save_time = datetime.now()

    def should_auto_save(self):
        if self.iteration % 25000 != 0:
            return

        if self.previous_auto_save_time is None:
            self.previous_auto_save_time = datetime.now()
            return False

        delta_auto_save_time = datetime.now() - self.previous_auto_save_time

        return delta_auto_save_time >= self.AUTO_SAVE_INTERVAL

    def has_initial_board_resulted(self):
        return self.get_board_result(self.initial_board) is not None

    def run_one(self):
        board = self.pop_board()
        if not board:
            return False

        try:
            self.try_run_one(board)
        except Exception:
            self.un_pop_board(board)
            raise

        return True

    def try_run_one(self, board):
        self.iteration += 1
        self.last_board = board

        next_boards = board.get_next_boards()
        if not next_boards:
            self.mark_board_result(board, Board.PLAYER_B)
            return

        any_next_board_winning = any(
            next_board
            for next_board in next_boards
            if next_board.has_player_b_won()
        )
        if any_next_board_winning:
            self.mark_board_result(board, Board.PLAYER_A)
            for next_board in next_boards:
                if next_board.has_player_b_won():
                    self.mark_board_result(next_board, Board.PLAYER_B)
                    self.mark_board_seen(next_board)
                self.add_sequence(board, next_board)
            return

        for next_board in next_boards:
            self.push_board(next_board, board)

        return

    def push_board(self, board, previous_board):
        if previous_board:
            self.add_sequence(previous_board, board)
        if self.mark_board_seen(board):
            self.push(board.equivalent_hash)
            self.queue_count += 1

    def pop_board(self):
        board = self.get_board_from_hash(self.pop())
        if board:
            self.queue_count -= 1
        else:
            self.queue_count = 0

        return board

    def un_pop_board(self, board):
        self.un_pop(board.equivalent_hash)
        self.queue_count += 1

    def mark_board_seen(self, board):
        if self.is_board_seen(board):
            return False
        self.mark_seen(board.equivalent_hash)
        self.seen_count += 1
        return True

    def is_board_seen(self, board):
        return self.is_seen(board.equivalent_hash)

    def add_sequence(self, previous_board, board):
        self.add_forward_board(previous_board, board)
        self.add_backward_board(previous_board, board)
        result = self.get_board_result(board)
        if result is not None:
            self.propagate_board_result_backwards(board, result)

    def add_forward_board(self, previous_board, board):
        self.add_forward(previous_board.equivalent_hash, board.equivalent_hash)

    def get_forward_boards(self, board):
        return [
            self.get_board_from_hash(equivalent_hash)
            for equivalent_hash in self.get_forward(board.equivalent_hash)
        ]

    def add_backward_board(self, previous_board, board):
        self.add_backward(previous_board.equivalent_hash, board.equivalent_hash)

    def get_backward_boards(self, board):
        return [
            self.get_board_from_hash(equivalent_hash)
            for equivalent_hash in self.get_backward(board.equivalent_hash)
        ]

    def get_board_from_hash(self, equivalent_hash):
        if not equivalent_hash:
            return None
        return self.board_type.from_equivalent_hash(equivalent_hash)

    def mark_board_result(self, board, result):
        if self.get_board_result(board) is not None:
            return

        self.mark_result(board.equivalent_hash, result)
        self.result_count += 1
        self.propagate_board_result_backwards(board, result)

    def propagate_board_result_backwards(self, board, result):
        if result == Board.PLAYER_B:
            for previous_board in self.get_backward_boards(board):
                self.mark_board_result(previous_board, Board.PLAYER_A)
        else:
            for previous_board in self.get_backward_boards(board):
                if self.get_board_result(previous_board) is not None:
                    continue
                all_forward_boards_lose = all(
                    self.get_board_result(next_board) == Board.PLAYER_A
                    for next_board in self.get_forward_boards(previous_board)
                )
                if all_forward_boards_lose:
                    self.mark_board_result(previous_board, Board.PLAYER_B)

    def get_board_result(self, board):
        return self.get_result(board.equivalent_hash)

    def mark_seen(self, equivalent_hash):
        raise NotImplementedError()

    def is_seen(self, equivalent_hash):
        raise NotImplementedError()

    def push(self, equivalent_hash):
        raise NotImplementedError()

    def pop(self):
        raise NotImplementedError()

    def un_pop(self):
        raise NotImplementedError()

    def add_forward(self, previous_hash, equivalent_hash):
        raise NotImplementedError()

    def get_forward(self, previous_hash):
        raise NotImplementedError()

    def add_backward(self, previous_hash, equivalent_hash):
        raise NotImplementedError()

    def get_backward(self, equivalent_hash):
        raise NotImplementedError()

    def mark_result(self, equivalent_hash, player):
        raise NotImplementedError()

    def get_result(self, equivalent_hash):
        raise NotImplementedError()


class MemoryQueue(BaseQueue):
    __slots__ = [
        'queue',
        'seen',
        'forward',
        'backward',
        'result_a',
        'result_b',
    ]
    def __init__(self, initial_board):
        super().__init__(initial_board)
        self.queue = None
        self.seen = None
        self.forward = None
        self.backward = None
        self.result_a = None
        self.result_b = None

    def get_state(self):
        state = super().get_state()

        state.update({
            'queue': tuple(self.queue) if self.queue is not None else None,
            'seen': tuple(self.seen) if self.seen is not None else None,
            'forward': self.forward,
            'backward': self.backward,
            'result_a': tuple(self.result_a) if self.result_a is not None else None,
            'result_b': tuple(self.result_b) if self.result_b is not None else None,
        })

        return state

    def apply_state(self, state, filename=None, resuming=False):
        super().apply_state(state, filename=filename, resuming=resuming)

        if state['queue'] is not None:
            self.queue = collections.deque(state['queue'])
        else:
            self.queue = None
        if state['seen'] is not None:
            self.seen = set(state['seen'])
        else:
            self.seen = None
        self.forward = state['forward']
        self.backward = state['backward']
        if 'result' in state:
            if state['result'] is not None:
                self.result_a = {
                    equivalent_hash
                    for equivalent_hash, player in state['result'].items()
                    if player == self.board_type.PLAYER_A
                }
                self.result_b = {
                    equivalent_hash
                    for equivalent_hash, player in state['result'].items()
                    if player == self.board_type.PLAYER_B
                }
            else:
                self.result_a = None
                self.result_b = None
        else:
            if state['result_a'] is not None:
                self.result_a = set(state['result_a'])
            else:
                self.result_a = None
            if state['result_b'] is not None:
                self.result_b = set(state['result_b'])
            else:
                self.result_b = None

    def clear_queue(self):
        self.queue = collections.deque()
        self.seen = set()
        self.forward = {}
        self.backward = {}
        self.result_a = set()
        self.result_b = set()

    def mark_seen(self, equivalent_hash):
        self.seen.add(equivalent_hash)

    def is_seen(self, equivalent_hash):
        return equivalent_hash in self.seen

    def push(self, equivalent_hash):
        self.queue.append(equivalent_hash)

    def pop(self):
        if not self.queue:
            return None
        return self.queue.popleft()

    def un_pop(self, equivalent_hash):
        self.queue.appendleft(equivalent_hash)

    def add_forward(self, previous_hash, equivalent_hash):
        self.forward.setdefault(previous_hash, []).append(equivalent_hash)

    def get_forward(self, previous_hash):
        return self.forward.get(previous_hash, [])

    def add_backward(self, previous_hash, equivalent_hash):
        self.backward.setdefault(equivalent_hash, []).append(previous_hash)

    def get_backward(self, equivalent_hash):
        return self.backward.get(equivalent_hash, [])

    def mark_result(self, equivalent_hash, player):
        if player is self.board_type.PLAYER_A:
            self.result_a.add(equivalent_hash)
        elif player is self.board_type.PLAYER_B:
            self.result_b.add(equivalent_hash)
        else:
            raise Exception("Unknown player '{}'".format(player))

    def get_result(self, equivalent_hash):
        if equivalent_hash in self.result_a:
            return self.board_type.PLAYER_A
        elif equivalent_hash in self.result_b:
            return self.board_type.PLAYER_B


class RedisQueue(BaseQueue):
    __slots__ = [
        'redis',
    ]

    def __init__(self, initial_board):
        super().__init__(initial_board)
        import redis
        self.redis = redis.Redis()

    def get_key(self, *names):
        return ":".join((self.initial_board.equivalent_hash,) + names)

    def get_key_queue(self):
        return self.get_key("queue")

    def get_key_seen(self):
        return self.get_key("seen")

    def get_key_forward(self, previous_hash):
        return self.get_key("forward", previous_hash)

    def get_key_backward(self, equivalent_hash):
        return self.get_key("backward", equivalent_hash)

    def get_key_result(self, equivalent_hash):
        return self.get_key("result", equivalent_hash)

    def _delete(self, key):
        self.redis.delete(key)

    def _set(self, key, value):
        self.redis.set(key, value)

    def _get(self, key):
        value = self.redis.get(key)
        if value is None:
            return None

        return value.decode()

    def _rpush(self, key, value):
        self.redis.rpush(key, value)

    def _lpush(self, key, value):
        self.redis.lpush(key, value)

    def _lpop(self, key):
        value = self.redis.lpop(key)
        if value is None:
            return None

        return value.decode()

    def _sadd(self, key, value):
        self.redis.sadd(key, value)

    def _sismember(self, key, value):
        self.redis.sismember(key, value)

    def _smembers(self, key):
        values = self.redis.smembers(key)
        return [value.decode() for value in values]

    def clear_queue(self):
        self._delete(self.get_key_queue())

    def mark_seen(self, equivalent_hash):
        self._sadd(self.get_key_seen(), equivalent_hash)

    def is_seen(self, equivalent_hash):
        return self._sismember(self.get_key_seen(), equivalent_hash)

    def push(self, equivalent_hash):
        self._rpush(self.get_key_queue(), equivalent_hash)

    def pop(self):
        return self._lpop(self.get_key_queue())

    def un_pop(self, equivalent_hash):
        self._lpush(self.get_key_queue(), equivalent_hash)

    def add_forward(self, previous_hash, equivalent_hash):
        self._sadd(self.get_key_forward(previous_hash), equivalent_hash)

    def get_forward(self, previous_hash):
        return self._smembers(self.get_key_forward(previous_hash))

    def add_backward(self, previous_hash, equivalent_hash):
        self._sadd(self.get_key_backward(equivalent_hash), previous_hash)

    def get_backward(self, equivalent_hash):
        return self._smembers(self.get_key_backward(equivalent_hash))

    def mark_result(self, equivalent_hash, player):
        self._set(self.get_key_result(equivalent_hash), player)

    def get_result(self, equivalent_hash):
        self._get(self.get_key_result(equivalent_hash))
