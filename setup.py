from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8-sig') as f:
    long_description = f.read()

# Read the contents of your requirements file
with open('README.md', 'r', encoding='utf-8-sig') as f:
    requirements = f.read().splitlines()

setup(
    name='customer_retention_toolkit',
    version='0.1',
    author='Tigran Boynagryan, Hayk Khachatryan, Vahagn Tovmasyan, Davit Davtyan, Elen Petrosyan',
    author_email='tigran.boynagryan@gmail.com',  # Replace with the actual contact email
    packages=find_packages(include=['customer_retention_tookit','customer_retention_toolkit.*']),
    description='A toolkit for customer retention analysis and prediction.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/davv13/DS223_Project',  # Replace with your actual repo URL
    license='MIT',
    keywords='customer retention machine learning',
)
