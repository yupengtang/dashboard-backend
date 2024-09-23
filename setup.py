import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="a21_server",
    version="0.0.1",
    description="Backend for the A21 Urban Planning project of the XAI-IML 2023 course.",
    #long_description=read("README.md"),
    package_data={
        "": [
            "parks_data.csv",
            "life_quality_indicators_chicago.csv"
        ]
    },
    data_files=[(
        "data", [
            os.path.join("data", "parks_data.csv"),
            os.path.join("data", "life_quality_indicators_chicago.csv"),
        ]
    )],
    entry_points={
        "console_scripts": [
            "start-server = app:start_server",
        ]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 2 - Pre-Alpha",
    ],
    install_requires=[
        "Flask>=2.0.0",
        "flask-restful>=0.3.9,<0.4",
        "flask-cors>=3.0.10,<3.1",
        "shap==0.41.0",
        "numba==0.56.4",
        # "plotly==5.8.0",
        "xgboost==1.6.2",
        "numpy==1.22.2",
        "pandas",
        "packaging",
        "flask_restful",
        "scikit-learn==1.0.2",
        "geopandas",
        "scipy",
        "matplotlib"
    ],
)
