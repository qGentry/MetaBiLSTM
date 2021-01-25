from setuptools import setup, find_packages


def main():
    """
    with open("requirements.txt") as f:
        requirements = f.read()
    """

    console_scripts = [
        "train_model = meta_bilstm.bin.train_model:main",
    ]

    setup(
        name="meta_bilstm",
        version="0.1",
        author="Philipp Fisin",
        package_dir={"": "src"},
        packages=find_packages("src"),
        description="Library for accenting word in questions",
        entry_points={
            "console_scripts": console_scripts
        }
    )


if __name__ == "__main__":
    main()
