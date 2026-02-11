"""Minimal RL configuration for scene viewing task."""

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def scene_view_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create minimal RL runner configuration for scene viewing.
    
    This is a minimal config since we're just viewing, not training.
    """
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,  # No observations needed for viewing
            critic_obs_normalization=False,
            actor_hidden_dims=(64, 32),  # Minimal network
            critic_hidden_dims=(64, 32),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=1,  # Minimal since not training
            num_mini_batches=1,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="scene_view",
        save_interval=1000,  # Not really used for viewing
        num_steps_per_env=24,
        max_iterations=100,  # Minimal
    )

