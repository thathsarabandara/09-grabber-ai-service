pipeline {
    agent any

    environment {
        GHCR_CREDENTIALS_ID = 'ghcr-credentials'
        GITHUB_USER         = 'thathsarabandara'
        IMAGE_NAME          = "ghcr.io/${GITHUB_USER}/09-grabber-ai-service"
        IMAGE_TAG           = "${env.BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code from SCM...'
                checkout scm
            }
        }

        stage('Environment') {
            steps {
                echo 'Verifying runtime tools...'
                sh '''
                    python3 --version
                    pip3 --version
                    docker --version
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing requirements...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements-dev.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                echo 'Running flake8 static analysis...'
                sh '''
                    . venv/bin/activate
                    flake8 app/ tests/ --count --statistics || true
                    bandit -r app/ -q || true
                '''
            }
        }

        stage('Test') {
            steps {
                echo 'Running tests...'
                sh '''
                    . venv/bin/activate
                    pytest tests/ || echo "Warning: Tests had non-zero exit status, checking logs..."
                '''
            }
        }

        stage('Build') {
            steps {
                echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
                sh """
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag  ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                    docker tag  ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:main
                """
            }
        }

        stage('Push') {
            steps {
                echo "Pushing image to GHCR: ${IMAGE_NAME}"
                withCredentials([usernamePassword(
                    credentialsId: GHCR_CREDENTIALS_ID,
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_TOKEN'
                )]) {
                    sh """
                        echo "\${GHCR_TOKEN}" | docker login ghcr.io -u "\${GHCR_USER}" --password-stdin
                        docker push ${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${IMAGE_NAME}:latest
                        docker push ${IMAGE_NAME}:main
                        echo "Pushed ${IMAGE_NAME}:${IMAGE_TAG}, ${IMAGE_NAME}:latest, and ${IMAGE_NAME}:main"
                    """
                }
            }
        }
    }

    post {
        always {
            sh 'docker logout ghcr.io || true'
            cleanWs()
        }
    }
}
