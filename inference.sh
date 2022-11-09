model_tag="kan-bayashi/jsut_tacotron2_accent_with_pause"
train_config="exp/tts_train_tacotron2_raw_phn_jaconv_none/config.yaml"
model_file="exp/tts_train_tacotron2_raw_phn_jaconv_none/180epoch.pth"

vocoder_tag="parallel_wavegan/jsut_hifigan.v1"
vocoder_config=""
vocoder_file=""

prosodic="false"
preg2p="false"
fs=""

reference=""

. utils/parse_options.sh

COMMAND="python inference.py "


pwg=`pip list | grep parallel`
if [ "$pwg" == "" ];
then
	pip install -U parallel_wavegan
fi

ip=`pip list | grep ipython`
if [ "$pwg" == "" ];
then
	pip install -U IPython
fi


if [ "$train_config" == "" ] && [ "$model_file" == "" ]
then
	COMMAND="${COMMAND}--model_tag \"${model_tag}\" "
else
	COMMAND="${COMMAND}--train_config \"${train_config}\" "
	COMMAND="${COMMAND}--model_file \"${model_file}\" "
fi

if [ "$vocoder_config" == "" ] && [ "$vocoder_file" == "" ]
then
	COMMAND="${COMMAND}--vocoder_tag \"${vocoder_tag}\" "
else
	COMMAND="${COMMAND}--vocoder_config \"${vocoder_config}\" "
	COMMAND="${COMMAND}--vocoder_file \"${vocoder_file}\" "
fi

if [ ! "$fs" == "" ]; then COMMAND="${COMMAND}--fs ${fs} "; fi

if [ "$prosodic" == "true" ]; then COMMAND="${COMMAND}-p "; fi
if [ "$preg2p" == "true" ]; then COMMAND="${COMMAND}--preg2p "; fi
if [ "$reference" == "true" ]; then COMMAND="${COMMAND}--reference"; fi

echo "${COMMAND}"
echo ""
echo ""

eval $COMMAND
