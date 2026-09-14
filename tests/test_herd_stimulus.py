"""A fly with no odour must not take the herd down with it.

`Olfaction.stimulation` returns no pulse when the genome's floor and width
leave every glomerulus at zero, and `FlyController.observe` drops it. The herd
did not, and an evolution of eighty-four random genomes died four minutes into
generation 0 on the first genome that produced none: `cannot unpack
non-iterable NoneType` inside `prepare_drive`.

No GPU and no brain here. The assembly is a plain function precisely so the
case can be checked without either.
"""

import numpy as np

from tools.evolve.herd import standing_pulses


class Channel:
    """Stands in for Olfaction or Gustation: returns whatever it is given."""

    def __init__(self, pulse):
        self.pulse = pulse

    def stimulation(self, *args):
        return self.pulse, "detail"


class Controller:
    def __init__(self, olfaction, gustation):
        self.olfaction = olfaction
        self.gustation = gustation


ODOR = (np.array([1, 2], np.int32), np.float32([3.0, 4.0]))
TASTE = (np.array([7], np.int32), np.float32(9.0))
ACCOUNT = ("100.0", "100.0")


def test_both_channels_speak():
    c = Controller(Channel(ODOR), Channel(TASTE))
    assert standing_pulses(c, [1.0], None, ACCOUNT) == [ODOR, TASTE]


def test_silent_odour_is_dropped_not_passed_on():
    c = Controller(Channel(None), Channel(TASTE))
    assert standing_pulses(c, [1.0], None, ACCOUNT) == [TASTE]


def test_silent_taste_is_dropped_too():
    c = Controller(Channel(ODOR), Channel(None))
    assert standing_pulses(c, [1.0], None, ACCOUNT) == [ODOR]


def test_both_silent_gives_an_empty_list_not_a_none():
    c = Controller(Channel(None), Channel(None))
    assert standing_pulses(c, [1.0], None, ACCOUNT) == []


def test_a_channel_that_is_not_fitted_is_not_asked():
    c = Controller(None, None)
    assert standing_pulses(c, [1.0], None, ACCOUNT) == []


def test_no_history_shuts_the_olfactory_channel():
    c = Controller(Channel(ODOR), Channel(TASTE))
    assert standing_pulses(c, None, None, ACCOUNT) == [TASTE]


def test_no_account_shuts_the_gustatory_channel():
    c = Controller(Channel(ODOR), Channel(TASTE))
    assert standing_pulses(c, [1.0], None, None) == [ODOR]
