"""Candidate six-dimensional observation group for CPU validation."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


def make_group(term_class):
    """Create the privileged-state group with an explicit left/right wheel order."""

    @configclass
    class PrivilegedStateCfg(ObsGroup):
        """Base motion, wheel contact flags, and normalized action-history slot."""

        state = ObsTerm(
            func=term_class,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_leg_4", "right_leg_4"],
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    return PrivilegedStateCfg()
