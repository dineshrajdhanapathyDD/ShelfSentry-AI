# ShelfSentry AI - container image for BOTH Lambda functions (arm64).
#
# One image, two entry points. CloudFormation sets each function's ImageConfig
# Command to select the handler:
#   API    -> shelfsentry.awslambda.api.handler
#   Worker -> shelfsentry.awslambda.worker.handler
#
# Why a container image (not a zip): the real OpenCV 5 native libraries
# (opencv-python==5.0.0.93 -> cv2 5.0.0) plus NumPy exceed Lambda's 250 MB
# unzipped zip limit. A container image allows up to 10 GB, so the REAL vision
# pipeline ships intact (no OpenCV 4 substitution, no mock path).
#
# Base image: AWS-managed Python 3.12 Lambda runtime on arm64 (Graviton). The
# aarch64 manylinux wheels for opencv-python/numpy resolve against it, so the
# build needs no system compiler.

FROM public.ecr.aws/lambda/python:3.12-arm64

# OpenCV's shared libraries need a few OS packages that are not in the slim
# Lambda base (libGL for cv2, glib for its runtime). Installed once at build.
RUN microdnf install -y mesa-libGL glib2 && microdnf clean all

# Install Python dependencies first (better layer caching). boto3 is added here
# explicitly: the application lazy-imports it for S3/DynamoDB/Bedrock/Lambda,
# and pinning it makes the image reproducible rather than relying on whatever
# the base image happens to bundle.
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
RUN python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Copy the application package.
COPY shelfsentry ${LAMBDA_TASK_ROOT}/shelfsentry

# Fail the BUILD if the running image cannot import the real OpenCV 5 module and
# report cv2.__version__ == 5.x. This makes "it built" mean "OpenCV 5 is present
# and importable on arm64", surfacing a packaging regression at build time
# rather than at first invocation.
RUN python -c "import cv2, numpy; v=cv2.__version__; print('cv2', v, 'numpy', numpy.__version__); assert v.split('.')[0]=='5', 'expected OpenCV 5, got '+v"

# Default handler is the API; the worker function overrides this via
# ImageConfig.Command in CloudFormation. Must be the dotted module path to the
# handler function, per the AWS Lambda Python runtime interface client.
CMD [ "shelfsentry.awslambda.api.handler" ]
