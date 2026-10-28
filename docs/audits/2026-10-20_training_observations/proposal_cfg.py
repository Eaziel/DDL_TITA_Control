"""Candidate additional observation groups; not installed into the task."""

from isaaclab.envs.mdp import height_scan
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


def make_groups(term_class, joint_names):
    """Create terrain and dynamics group configs for CPU validation."""

    @configclass
    class ScanCfg(ObsGroup):
        """Terrain heights relative to the nominal base clearance."""

        heights = ObsTerm(
            func=height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.4},
            clip=(-1.0, 1.0),
            scale=5.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DynamicsCfg(ObsGroup):
        """Randomized physical parameters available during training."""

        parameters = ObsTerm(
            func=term_class,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names="base_link", joint_names=joint_names, preserve_order=True
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    return ScanCfg(), DynamicsCfg()
