# pipeline deploy react

```json
pipeline {
    environment {
        SERVER_EC2_1 = "13.113.53.107"
        SERVER_DEPLOY_DIR = "/home/ec2-user/www/"
        SERVER_CREDENTIALSID = "bf16ceb9-5cfd-4f05-b9eb-978fcb4c8b87"

        CACHE_DIR = "/var/jenkins_home/wlp-client-react/"

        GIT_URL = "https://gitlab.ominext.co/k_board/wlp-client-react.git"
        GIT_BRANCH = "jenkins-test"
        GIT_CREDENTIALSID = "fb459f3e-da9e-4556-b76a-a0be5e7263d6"
    }
    agent none
    stages {
        stage('Checkout code') {
            agent any
            steps {
                git (
                    branch: "${GIT_BRANCH}",
                    credentialsId: "${GIT_CREDENTIALSID}",
                    url: "${GIT_URL}",
                    changelog: true
                )
                sh '''
                    ls -al
                    cache_dir="${CACHE_DIR}/"
                    cache_nm="${CACHE_DIR}/node_modules"

                    if [ ! -d "$cache_dir" ]; then mkdir ${cache_dir}; fi
                    if [ ! -d "$cache_nm" ]; then mkdir ${cache_nm}; fi
                    if [ -d "$cache_nm" ]; then ln -sf ${cache_nm} ./; fi

                    ls -al
                '''
            }
        }
        stage('Build') {
            agent {
                docker {
                    image 'node:13-alpine'
                    args ''
                }
            }
            steps {
                sh '''
					ls -al
					npm install
					node -v
                    npm run build:mypage
					cd build/
                    tar -cvf mypage.tar mypage

                    ls -al
					cd ..
                    rm -rf ./node_modules
                    ls -al
                '''
                archiveArtifacts artifacts: 'build/mypage.tar', fingerprint: true
            }
        }
        stage('Deploy') {
            agent any
            steps {
                unarchive mapping: ['build/mypage.tar': 'build/mypage.tar']
                echo '--- Deploy ---'

				sshagent(credentials : ["${SERVER_CREDENTIALSID}"]) {
					sh 'ssh -o StrictHostKeyChecking=no ec2-user@${SERVER_EC2_1} uptime'
					sh 'ssh -v ec2-user@${SERVER_EC2_1}'
					sh 'scp build/mypage.tar ec2-user@${SERVER_EC2_1}:${SERVER_DEPLOY_DIR}'
					sh 'ssh ec2-user@${SERVER_EC2_1} tar -xvf ${SERVER_DEPLOY_DIR}mypage.tar -C ${SERVER_DEPLOY_DIR}reactjs'


				}

            }
        }
    }
}
```