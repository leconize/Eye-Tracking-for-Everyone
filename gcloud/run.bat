SET datetime=%date:~-10,2%%date:~-7,2%%date:~-4,4%_%time:~0,2%%time:~3,2%%time:~6,2%
call set datetime=%datetime: =%

set BUCKET_NAME=gazeasia
set JOB_NAME="talking_data_lstm_%datetime%"
set JOB_DIR=gs://%BUCKET_NAME%/keras-train
set REGION=asia-east1
set CONFIG=config.yaml
gcloud ml-engine jobs submit training %JOB_NAME% ^
--job-dir %JOB_DIR% ^
--runtime-version 1.6 ^
--config %CONFIG% ^
--module-name trainer.gaze_trainer ^
--package-path ./trainer ^
--region %REGION% ^
-- ^
--train-file gs://%BUCKET_NAME%/binary_input/pickle