# STEP 1 - Create ACI Deployment Configuration:

##Import necessary Azure ML classes

from azureml.core.webservice import AciWebservice, Webservice 
from azureml.core.model import InferenceConfig

##Define the ACI configuration, specifying CPU and memory:
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)


# STEP 2 - Set Up Inference Configuration:

##Create an inference configuration using your score.py and environment.yml:
inference_config = InferenceConfig(runtime= "python", entry_script="score.py", conda_file="environment.yml") 


# STEP 3 - Deploy the Model:

##Deploy the model using the Azure ML SDK:
from azureml.core import Workspace
from azureml.core.model import Model

import model_deployment

service = Model.deploy(workspace=workspace, name="rocky", models=[model], inference_config=inference_config, deployment_config=aci_config)
service.wait_for_deployment(show_output=True)