"""AWS Lambda entry points for ShelfSentry AI (container-image deployment).

This package is the thin transport layer that adapts AWS Lambda events to the
mode-agnostic core. It contains NO perception, decision, or persistence logic of
its own - it only:

* builds an :class:`~shelfsentry.data.aws_storage.AwsStorageAdapter` from the
  non-secret ``SHELFSENTRY_*`` environment variables the CloudFormation stack
  sets (Req 16.1, 17.2); and
* routes an incoming event to the existing entry points
  (:class:`~shelfsentry.api.handlers.LocalApi` for the synchronous API;
  :func:`shelfsentry.aws_worker.run_async_job` for the async worker).

Both handlers run the REAL OpenCV 5 vision pipeline and the REAL bounded agent
loop with Amazon Bedrock as the primary decision path; there is no mock/placeholder
result path here (Req 16.4).
"""
