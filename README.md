# deeplab_v2
基于v2版本的deeplab,使用VGG16模型，在VOC2012，Pascal-context，NYU-v2等多个数据集上进行训练


好记性不如烂笔头, 最近用Deeplab v2跑的图像分割，现记录如下。
官方源码地址如下：https://bitbucket.org/aquariusjay/deeplab-public-ver2/overview 
但是此源码只是为deeplab网络做相应变形的caffe,如果需要fine tuning微调网络，还需要准备以下文件：

 - **txt文件**：文件中有数据集的名字列表的txt文件,[训练测试集列表](https://ucla.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb)
 
 - **训练好的init.caffemodel**: 针对deeplab v2，作者有已经预训练好的两个模型参数：[DeepLabv2_VGG16 ](http://liangchiehchen.com/projects/released/deeplab_aspp_vgg16/prototxt_and_model.zip)和[DeepLabv2_ResNet101
](http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip) 

 - **网络结构prototxt文件**: train.prototxt和solver.prototxt，分别在：[DeepLabv2_VGG16 ](http://liangchiehchen.com/projects/released/deeplab_aspp_vgg16/prototxt_and_model.zip)和 [DeepLabv2_ResNet101
](http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip) 
 - **官网脚本文件**: [三个sh文件](https://ucla.box.com/s/4grlj8yoodv95936uybukjh5m0tdzvrf)，建议使用脚本文件，初看虽不懂，但是比[python版本](https://github.com/TheLegendAli/CCVL)的运行简单很多
注：本博客只涉及脚本版本的训练

----------
### 1.建立deeplab文件夹, 并下载deeplab源代码


```
cd ~
mkdir deeplab
cd deeplab
git clone https://bitbucket.org/aquariusjay/deeplab-public-ver2.git
```
### 2.依次建立存放设置文件夹，预测结果文件夹，数据集txt文件夹，log文件夹，model文件夹，evaluation文件夹

```
mkdir  -p ~/deeplab/exper/voc12/config/deeplab_largeFOV
mkdir  -p ~/deeplab/exper/voc12/features/labels
mkdir -p ~/deeplab/exper/voc12/features2/labels
mkdir -p ~/deeplab/exper/voc12/list
mkdir -p ~/deeplab/exper/voc12/log 
mkdir -p ~/deeplab/exper/voc12/model/deeplab_largeFOV
mkdir -p ~/deeplab/exper/voc12/res

```
### 3.下载官方给的txt文件夹，以及预训练的model和网络结构文件，如上所示。
有时候可能会打不开网页，无法访问，也可以在我的资源中下载，我已经原资料打包上传，[无法在官网下载就点这里](http://download.csdn.net/download/xmo_jiao/9943695)
如下：
![这里写图片描述](http://img.blog.csdn.net/20170831174307230?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


![这里写图片描述](http://img.blog.csdn.net/20170831174324237?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


 ![这里写图片描述](http://img.blog.csdn.net/20170831174551587?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


### 4. 把下载解压后的相应文件移动到相应文件夹

以prototxt为后缀的网络结构文件train.prototxt 、test.prototxt 以及solver.prototxt文件移动到~/deeplab/exper/voc12/config/deeplab_largeFOV
文件夹下.

```
unzip prototxt_and_model.zip
mv *.prototxt ~/deeplab/exper/voc12/config/deeplab_largeFOV
mv *.caffemodel ~/deeplab/exper/voc12/model/deeplab_largeFOV
unzip link.zip
cd link
mv * ~/deeplab/exper/voc12
unzip list.zip
cd list
mv * ~/deeplab/exper/voc12/list

```
### 5.数据集处理
### 6.脚本文件修改


-------------------

### deeplab2的script文件run_pascal.sh 解析
```
#!/bin/sh 

## MODIFY PATH for YOUR SETTING
ROOT_DIR= 

CAFFE_DIR=../code #你的caffe路径，clone作者的deeplab v2得到deeplab-public-ver2文件夹，即为此处caffe路径， 注意：此处caffe要编译
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin 

EXP=voc12 #此目录路径~/deeplab/exper/voc12

if [ "${EXP}" = "voc12" ]; then
    NUM_LABELS=21
    DATA_ROOT=${ROOT_DIR}/rmt/data/pascal/VOCdevkit/VOC2012 #VOC数据目录，修改为你的数据目录 
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi
 

## Specify which model to train
########### voc12 ################
NET_ID=deelab_largeFOV  ##此处文件名有问题应该改为deeplab_largeFOV


## Variables used for weakly or semi-supervisedly training
#TRAIN_SET_SUFFIX=
#TRAIN_SET_SUFFIX=_aug   #此处应该取消注释，当你run training 1时

#TRAIN_SET_STRONG=train  
#TRAIN_SET_STRONG=train200
#TRAIN_SET_STRONG=train500
#TRAIN_SET_STRONG=train1000
#TRAIN_SET_STRONG=train750

#TRAIN_SET_WEAK_LEN=5000

DEV_ID=0

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID} #此处目录为/voc12/config/deeplab_largeFOV
MODEL_DIR=${EXP}/model/${NET_ID} 
mkdir -p ${MODEL_DIR} #创建MODEL_DIR目录为/voc12/model/deeplab_largeFOV
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAIN=1 #为1说明执行train
RUN_TEST=1  #为1说明执行test
RUN_TRAIN2=0
RUN_TEST2=0

## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then  #r如果RUN_TRAIN为1
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then #如果TRAIN_SET_WEAK_LEN长度为零则为真
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt #comm -3 指令为不输出两个文件共有的行，此处即为除去train.txt文件中train_aug.txt的数据，其他都输出到train_aud_diff_train.txt
    else
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init.caffemodel #下载的vgg16或者ResNet101中的 model
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt #复制文件train.prototxt到train_train_train_aug.prototxt,slove同理
    done #此部分运行时如以下命令
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
		if [ -f ${MODEL} ]; then
				CMD="${CMD} --weights=${MODEL}"
		fi
		echo Running ${CMD} && ${CMD}  
fi
#train部分运行时，即以下运行命令 ../deeplab-public-ver2/.build_release/tools/caffe.bin train --solver=volab_largeFOV/solver_train_aug.prototxt --gpu=0 --weights=voc12/model/deeplab_largeFOV/init.caf   femodel
#上述命令中，solver_train_aug.prototxt由solve.prototxt文件复制而来，init.caffemodel为原始下载了的VGG16的model
## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l` #此处计算val.txt文件中测试图片个数，共1449个
				MODEL=${EXP}/model/${NET_ID}/test.caffemodel
				if [ ! -f ${MODEL} ]; then
						MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
				fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
        mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi
#test部分运行时，即以下运行命令../deeplab-public-ver2/.build_release/tools/caffe.bin test --model=voc12/config/deeplab_largeFOV/test_val.prototxt --weights=voc12/model/deeplab_largeFOV/train_iter_20000.caffemodel --gpu=0 --iterations=1449
#上述命令中，test_val.prototxt由test.prototxt文件复制而来，train_iter_20000.caffemode由第一部分train得到的model
## Training #2 (finetune on trainval_aug)

if [ ${RUN_TRAIN2} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=trainval${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init2.caffemodel
    if [ ! -f ${MODEL} ]; then
				MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
    fi
    #
    echo Training2 net ${EXP}/${NET_ID}
    for pname in train solver2; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
    CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver2_${TRAIN_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID}"
		echo Running ${CMD} && ${CMD}
fi

## Test #2 on official test set

if [ ${RUN_TEST2} -eq 1 ]; then
    #
    for TEST_SET in val test; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				MODEL=${EXP}/model/${NET_ID}/test2.caffemodel
				if [ ! -f ${MODEL} ]; then
						MODEL=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.caffemodel | head -n 1`
				fi
				#
				echo Testing2 net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features2/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi

```





---------

