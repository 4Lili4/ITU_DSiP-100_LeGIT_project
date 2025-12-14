#DEPLOY ___________________________
#A model version can be assigned to one or more stages. MLflow provides predefined stages for common use cases: None, Staging, Production, and Archived. With the necessary permissions, you can transition a model version between stages or request a transition to a different stage.

from mlflow.tracking import MlflowClient


def wait_for_deployment(model_name, model_version, stage='Staging'):

    #set status to false to initiate a while-loop
    status = False
    while not status:

        #fetch model details
        model_version_details = dict(client.get_model_version(name=model_name,version=model_version))

        #if the model-details show the model is now in 'Staging', the loop is broken and the transition is complete
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break

        #If the model-details show that the model has not yet transitioned to 'staging', program sleeps for 2 counts and retries
        else:
            time.sleep(2)

    #Will return True
    return status

#Function 
def deploy_to_staging(model_name,
                      model_version = 1,
                      client = MlflowClient()):

    #Fetch model information
    model_version_details = dict(client.get_model_version(name=model_name,version=model_version))

    model_status = True

    #If the model is in another stage than 'Staging', this and the following line initialise the transition
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(name=model_name, version=model_version,stage="Staging", archive_existing_versions=True)

        #Call wait_for_deployment function s.t. the line "Transition completed to 'Staging' is not printed until the model is actually updated
        model_status = wait_for_deployment(model_name, model_version, 'Staging')

    #Of course, if the model is already in staging, this function should do nothing but print a user-friendly message
    else:
        print('Model already in staging')