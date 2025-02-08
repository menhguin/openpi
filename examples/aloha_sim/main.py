import dataclasses
import enum
import logging
import pathlib

from custom_env import CustomAlohaEnv
import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


class EnvType(enum.Enum):
    DEFAULT = "default"
    CUSTOM = "custom"


@dataclasses.dataclass
class Args:
    # Environment settings
    env_type: EnvType = EnvType.DEFAULT
    task: str = "gym_aloha/AlohaTransferCube-v0"  # Only used for default env
    
    # Output settings
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")
    seed: int = 0

    # Policy settings
    action_horizon: int = 10
    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False


def create_environment(args: Args):
    """Create the appropriate environment based on args."""
    if args.env_type == EnvType.DEFAULT:
        return _env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
        )
    elif args.env_type == EnvType.CUSTOM:
        return CustomAlohaEnv(seed=args.seed)
    else:
        raise ValueError(f"Unknown environment type: {args.env_type}")


def main(args: Args) -> None:
    # Create the environment
    env = create_environment(args)
    
    # Create and run the runtime
    runtime = _runtime.Runtime(
        environment=env,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            ),
        ),
        subscribers=[_saver.VideoSaver(args.out_dir)],
        max_hz=50,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
