from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="rl-tuning",
    version="0.1",
    description="Reinforcement Learning Agents from UoE RL Course 2023",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)
