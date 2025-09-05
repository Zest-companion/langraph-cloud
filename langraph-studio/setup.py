from setuptools import setup, find_packages

setup(
    name="langraph-studio",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["zest_companion_workflow_modular", "zest_companion_workflow", "zest_mbti_workflow"],
    include_package_data=True,
)