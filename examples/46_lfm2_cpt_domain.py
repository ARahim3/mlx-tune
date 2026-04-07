"""
Example 46: LFM2 + Continual Pretraining (Combined)

Demonstrates using LFM2 models with CPT for domain adaptation.
Combines LFM2-specific target modules with embedding training.

LFM2's efficient hybrid architecture (gated convolutions + GQA)
makes it ideal for CPT on domain corpora — fast training with
good knowledge retention.

NOTE: Requires downloading LFM2 model.
"""

from mlx_tune import FastLanguageModel, CPTTrainer, CPTConfig
from datasets import Dataset


def main():
    print("=" * 70)
    print("LFM2 + Continual Pretraining - Domain Adaptation")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load LFM2 Base Model
    # ========================================================================
    print("\n[Step 1] Loading LFM2 model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/LFM2-350M-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA with LFM2 Targets
    # ========================================================================
    print("\n[Step 2] Applying LoRA with LFM2-specific targets...")

    # LFM2 uses in_proj/out_proj (attention) and w1/w2/w3 (gated conv MLP)
    # CPTTrainer will auto-add embed_tokens + lm_head
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "in_proj", "w1", "w2", "w3",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    # ========================================================================
    # Step 3: Prepare Domain Corpus (Robotics/Automation)
    # ========================================================================
    print("\n[Step 3] Preparing robotics domain corpus...")

    robotics_corpus = [
        {"text": "Robotic manipulation requires precise control of end-effector position and orientation in three-dimensional space. Inverse kinematics algorithms compute the joint angles needed to achieve a desired pose. The Jacobian matrix relates joint velocities to end-effector velocities, enabling real-time trajectory tracking. Redundant manipulators with more degrees of freedom than task dimensions offer additional optimization objectives such as obstacle avoidance and joint limit avoidance."},
        {"text": "Simultaneous Localization and Mapping (SLAM) enables autonomous robots to build a map of an unknown environment while simultaneously tracking their position within it. Visual SLAM uses camera images to extract features like ORB or SIFT keypoints for loop closure detection. LiDAR-based SLAM methods like LOAM provide precise 3D point cloud maps. Graph-based optimization techniques refine the estimated trajectory by minimizing the error between predicted and observed landmarks."},
        {"text": "Reinforcement learning for robotics faces the sim-to-real transfer challenge. Policies trained in simulation often fail when deployed on physical robots due to the reality gap. Domain randomization addresses this by training across varied simulation parameters including friction, mass, lighting, and sensor noise. Curriculum learning gradually increases task difficulty, enabling robots to learn complex behaviors like dexterous manipulation and bipedal locomotion from simpler sub-tasks."},
        {"text": "Robot Operating System (ROS 2) provides a middleware framework for building robot applications. The publish-subscribe communication model enables loosely coupled nodes to exchange messages over topics. Services provide synchronous request-response interaction, while actions handle long-running tasks with feedback. The tf2 library manages coordinate frame transformations essential for multi-sensor fusion and robot navigation."},
        {"text": "Force-torque sensing enables compliant robot manipulation for assembly tasks and human-robot collaboration. Impedance control modulates the robot's mechanical impedance to achieve desired interaction dynamics. In peg-in-hole insertion tasks, the robot must detect contact forces and adapt its trajectory to accommodate positional uncertainty. Hybrid force-position control splits the task space into force-controlled and position-controlled directions."},
        {"text": "Autonomous mobile robots navigate using a layered architecture of global and local planners. A* and RRT* algorithms compute collision-free paths on occupancy grid maps. The Dynamic Window Approach (DWA) generates velocity commands that avoid obstacles while tracking the global path. Multi-robot coordination requires distributed planning to prevent collisions and optimize task allocation across the fleet."},
        {"text": "Computer vision for robotics extends beyond object detection to include 6-DoF pose estimation, semantic segmentation, and depth prediction. PointNet architectures process 3D point clouds directly for grasping pose estimation. Foundation models like SAM enable zero-shot segmentation for novel objects. Visual servoing closes the loop between perception and control by directly using image features as control inputs."},
        {"text": "Soft robotics uses compliant materials like silicone elastomers and shape memory alloys to create robots that safely interact with humans and delicate objects. Pneumatic artificial muscles provide high force-to-weight ratios through pressurized air chambers. Finite element methods model the nonlinear deformation of soft bodies. Bio-inspired designs mimic octopus tentacles and elephant trunks for dexterous manipulation without rigid joints."},
    ]

    dataset = Dataset.from_list(robotics_corpus)
    print(f"Dataset: {len(dataset)} robotics documents")

    # ========================================================================
    # Step 4: Train with CPT
    # ========================================================================
    print("\n[Step 4] Starting CPT with LFM2...")

    config = CPTConfig(
        learning_rate=5e-5,
        embedding_learning_rate=5e-6,
        include_embeddings=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        logging_steps=5,
        output_dir="outputs_lfm2_cpt",
        max_seq_length=2048,
    )

    trainer = CPTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    trainer.train()

    # ========================================================================
    # Step 5: Test Domain Knowledge
    # ========================================================================
    print("\n[Step 5] Testing robotics knowledge...")

    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompts = [
        "Simultaneous Localization and Mapping (SLAM) enables",
        "Reinforcement learning for robotics",
    ]

    for prompt in prompts:
        response = generate(
            model.model, tokenizer,
            prompt=prompt,
            max_tokens=80,
            verbose=False,
        )
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {response[:150]}")

    print("\nLFM2 + CPT domain adaptation complete!")
    print("\nLFM2's efficient architecture makes it ideal for CPT:")
    print("  - Hybrid conv+attention processes sequences efficiently")
    print("  - 2.6B params fits in Mac memory even with embeddings unfrozen")
    print("  - Fast inference after adaptation for edge deployment")


if __name__ == "__main__":
    main()
