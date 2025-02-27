#!/bin/bash

# Netcore AI Marketing Suite Deployment Script
# This script builds and deploys the application to cloud providers

# Configuration
IMAGE_NAME="netcore-marketing-suite"
VERSION=$(git describe --tags --always --dirty || echo "0.1.0")
FULL_IMAGE_NAME="${IMAGE_NAME}:${VERSION}"

# Build the Docker image
echo "Building Docker image ${FULL_IMAGE_NAME}..."
docker build -t ${FULL_IMAGE_NAME} .

# Function for AWS deployment
deploy_to_aws() {
    echo "Deploying to AWS..."
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    # Create repository if it doesn't exist
    aws ecr describe-repositories --repository-names ${IMAGE_NAME} || aws ecr create-repository --repository-name ${IMAGE_NAME}
    
    # Tag and push image
    ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
    docker tag ${FULL_IMAGE_NAME} ${ECR_REPO}:${VERSION}
    docker tag ${FULL_IMAGE_NAME} ${ECR_REPO}:latest
    docker push ${ECR_REPO}:${VERSION}
    docker push ${ECR_REPO}:latest
    
    # Update ECS service (if using ECS)
    if [ "$1" == "ecs" ]; then
        aws ecs update-service --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --force-new-deployment
    fi
    
    # Deployment instructions for Lambda
    if [ "$1" == "lambda" ]; then
        echo "To deploy to Lambda, run:"
        echo "aws lambda update-function-code --function-name ${LAMBDA_FUNCTION} --image-uri ${ECR_REPO}:${VERSION}"
    fi
}

# Function for GCP deployment
deploy_to_gcp() {
    echo "Deploying to Google Cloud..."
    
    # Configure Docker to use gcloud as a credential helper
    gcloud auth configure-docker
    
    # Tag and push image
    GCR_REPO="gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}"
    docker tag ${FULL_IMAGE_NAME} ${GCR_REPO}:${VERSION}
    docker tag ${FULL_IMAGE_NAME} ${GCR_REPO}:latest
    docker push ${GCR_REPO}:${VERSION}
    docker push ${GCR_REPO}:latest
    
    # Deploy to Cloud Run
    if [ "$1" == "cloud-run" ]; then
        gcloud run deploy ${IMAGE_NAME} --image ${GCR_REPO}:${VERSION} --platform managed --region ${GCP_REGION} --allow-unauthenticated
    fi
}

# Parse command line arguments
case "$1" in
    aws-ecs)
        AWS_REGION=${2:-"us-east-1"}
        AWS_ACCOUNT_ID=${3:-$(aws sts get-caller-identity --query Account --output text)}
        ECS_CLUSTER=${4:-"netcore-cluster"}
        ECS_SERVICE=${5:-"netcore-service"}
        deploy_to_aws "ecs"
        ;;
    aws-lambda)
        AWS_REGION=${2:-"us-east-1"}
        AWS_ACCOUNT_ID=${3:-$(aws sts get-caller-identity --query Account --output text)}
        LAMBDA_FUNCTION=${4:-"netcore-function"}
        deploy_to_aws "lambda"
        ;;
    gcp-cloud-run)
        GCP_PROJECT_ID=${2:-$(gcloud config get-value project)}
        GCP_REGION=${3:-"us-central1"}
        deploy_to_gcp "cloud-run"
        ;;
    *)
        echo "Usage:"
        echo "  $0 aws-ecs [region] [account-id] [cluster] [service]"
        echo "  $0 aws-lambda [region] [account-id] [function-name]"
        echo "  $0 gcp-cloud-run [project-id] [region]"
        exit 1
        ;;
esac

echo "Deployment completed!" 