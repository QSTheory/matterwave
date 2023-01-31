import setuptools
import subprocess

version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

assert "." in version

setuptools.setup(
    name="matterwave",
    version=version,
    author="Stefan Seckmeyer, Gabriel Müller, Christian Struckmann",
    author_email="",
    description="",
    long_description="",
    long_description_content_type="text/plain",
    url="",
    packages=["matterwave"],
    include_package_data=True,
    package_data={"matterwave": ["py.typed"]},
    zip_safe=False,
    classifiers=[],
    python_requires=">=3.9",
    install_requires=[
        "fftarray",
        "numpy>=1.21",
        "scipy",
        "jax>=0.4.2",
        "jaxlib",
    ],
    extras_require={
        "dev": [
            "mypy>=0.910",
            "pytest",
            "hypothesis",
            "sphinx>=4.2",
            "sphinx_rtd_theme",
            "myst_parser",
            "mistune==0.8.4",
            "m2r2"
        ],
    }

)
