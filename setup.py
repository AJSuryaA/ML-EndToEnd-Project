from setuptools import find_packages, setup

def get_requsites(file_path:str)->list[str]:
    '''
    this fun is to get list of requsites from requirements.txt
    '''
    requsites= []
    with open(file_path) as file:
        requsites= file.readlines()
        requsites= [req.replace("\n", " ") for req in requsites]
        if "-e ." in requsites:
            requsites.remove("-e .")
    return requsites

setup(
    name= 'ML EndToEnd Project',
    version= '0.0.1',
    author= 'JAYA SURYA A',
    author_email= 'surya.2003amuari@gmail.com',
    packages= find_packages(),
    install_requisites= get_requsites('requirements.txt')
)