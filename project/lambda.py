import json
import boto3
import base64
import os
import io
import csv

s3 = boto3.client('s3')
def SerializeImageData(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        byte_data= f.read()
        image_data = base64.b64encode(byte_data)

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2021-12-06-19-23-39-645'
runtime= boto3.client('runtime.sagemaker')

def ClassifyImageData(event, context):
    
    # Decode the image data
    data = json.loads(json.dumps(event))
    image = base64.b64decode(data['body']['image_data'])
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                        Body=image)
    
    inferences = json.loads(response['Body'].read().decode())

    event["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': {'inferences':inferences}
    }
    
    import json


THRESHOLD = .93
def FilterImageData(event, context):

    # Grab the inferences from the event
    data = json.loads(json.dumps(event))
    inferences = data['body']['inferences']

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = True in (float(i) > .93 for i in inferences)
    print('meets_threshold', meets_threshold)
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }