#pragma once
#include <cstdint>

struct yolo_template_opts {
	// v7 defaults
	size_t   width        = 640;
	size_t   height       = 640;
	size_t   batch        = 8;
	size_t   subdivisions = 1;
	size_t   channels     = 3;
	size_t   classes      = 32; // number of objects you'd like to detect!
	uint32_t mixup        = 1;  // we want to

	size_t max_batches;   // std::max(std::max(classes*2000, 6000), training_images_count);
	size_t line_steps_80; // (max_batches * 8) / 10;
	size_t line_steps_90; // (max_batches * 9) / 10;
	size_t filters;       // (classes + 5) * 3
};

constexpr std::string_view yolov7_template = R"(
[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch={batch}
subdivisions={subdivisions}
width={width}
height={height}
mixup={mixup}
channels={channels}
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000

max_batches = {max_batches}
policy=steps
steps={line_steps_80},{line_steps_90}
scales=.1,.1

# 0
[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=swish


# 1
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish


# 3
[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-3,-5,-7

# 12
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish


[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[route]
layers=-3

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=swish

# 18
[route]
layers = -1,-4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-3,-5,-7

# 27
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish


[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[route]
layers=-3

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=swish

# 33
[route]
layers = -1,-4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-3,-5,-7

# 42
[convolutional]
batch_normalize=1
filters=1024
size=1
stride=1
pad=1
activation=swish


[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

[route]
layers=-3

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=swish

# 48
[route]
layers = -1,-4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-3,-5,-7

# 57
[convolutional]
batch_normalize=1
filters=1024
size=1
stride=1
pad=1
activation=swish

##################################

### SPPCSP ###
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

[route]
layers = -2

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=swish

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

### SPP ###
[maxpool]
stride=1
size=5

[route]
layers=-2

[maxpool]
stride=1
size=9

[route]
layers=-4

[maxpool]
stride=1
size=13

[route]
layers=-6,-5,-3,-1
### End SPP ###

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=swish

[route]
layers = -1, -13

# 72
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish
### End SPPCSP ###


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[upsample]
stride=2

[route]
layers = 42

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[route]
layers = -1,-3


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-2,-3,-4,-5,-7

# 86
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[upsample]
stride=2

[route]
layers = 27

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[route]
layers = -1,-3


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-2,-3,-4,-5,-7

# 100
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish


[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[route]
layers=-3

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=swish

[route]
layers = -1,-4,86


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-2,-3,-4,-5,-7

# 115
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish


[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[route]
layers=-3

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=swish

[route]
layers = -1,-4,72


[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=swish

[route]
layers = -1,-2,-3,-4,-5,-7

# 130
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=swish

#############################

# ============ End of Neck ============ #

# ============ Head ============ #


# P3
[route]
layers = 100

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=swish

[convolutional]
size=1
stride=1
pad=1
filters={filters}
#activation=linear
activation=logistic

[yolo]
mask = 0,1,2
anchors = 12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401
classes={classes}
num=9
jitter=.1
scale_x_y = 2.0
objectness_smooth=1
ignore_thresh = .7
truth_thresh = 1
#random=1
resize=1.5
iou_thresh=0.2
iou_normalizer=0.05
cls_normalizer=0.5
obj_normalizer=1.0
iou_loss=ciou
nms_kind=diounms
beta_nms=0.6
new_coords=1
max_delta=2


# P4
[route]
layers = 115

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=swish

[convolutional]
size=1
stride=1
pad=1
filters={filters}
#activation=linear
activation=logistic

[yolo]
mask = 3,4,5
anchors = 12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401
classes={classes}
num=9
jitter=.1
scale_x_y = 2.0
objectness_smooth=1
ignore_thresh = .7
truth_thresh = 1
#random=1
resize=1.5
iou_thresh=0.2
iou_normalizer=0.05
cls_normalizer=0.5
obj_normalizer=1.0
iou_loss=ciou
nms_kind=diounms
beta_nms=0.6
new_coords=1
max_delta=2


# P5
[route]
layers = 130

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=swish

[convolutional]
size=1
stride=1
pad=1
filters={filters}
#activation=linear
activation=logistic

[yolo]
mask = 6,7,8
anchors = 12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401
classes={classes}
num=9
jitter=.1
scale_x_y = 2.0
objectness_smooth=1
ignore_thresh = .7
truth_thresh = 1
#random=1
resize=1.5
iou_thresh=0.2
iou_normalizer=0.05
cls_normalizer=0.5
obj_normalizer=1.0
iou_loss=ciou
nms_kind=diounms
beta_nms=0.6
new_coords=1
max_delta=2
)";

constexpr std::string_view yolov7_tiny_template =
				R"([net]
# Testing
#batch=1
#subdivisions=1
# Training
batch={batch}
subdivisions={subdivisions}
width={width}
height={height}
mixup={mixup}
channels={channels}
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000

max_batches = {max_batches}
policy=steps
steps={line_steps_80},{line_steps_90}
scales=.1,.1

# 0
[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

# 1
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 8
[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 16
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 24
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 32
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky


##################################

### SPPCSP ###
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

### SPP ###
[maxpool]
stride=1
size=5

[route]
layers=-2

[maxpool]
stride=1
size=9

[route]
layers=-4

[maxpool]
stride=1
size=13

[route]
layers=-1,-3,-5,-6
### End SPP ###

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -10,-1

# 44
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky
### End SPPCSP ###

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = 24

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1,-3

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 56
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = 16

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -1,-3

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 68
[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

##########################

[convolutional]
batch_normalize=1
size=3
stride=2
pad=1
filters=128
activation=leaky

[route]
layers = -1,56

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 77
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=2
pad=1
filters=256
activation=leaky

[route]
layers = -1,44

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -5,-3,-2,-1

# 86
[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

#############################

# ============ End of Neck ============ #

# ============ Head ============ #


# P3
[route]
layers = 68

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=128
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
#activation=linear
activation=logistic

[yolo]
mask = 0,1,2
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
classes={classes}
num=9
jitter=.1
scale_x_y = 2.0
objectness_smooth=1
ignore_thresh = .7
truth_thresh = 1
#random=1
resize=1.5
iou_thresh=0.2
iou_normalizer=0.05
cls_normalizer=0.5
obj_normalizer=1.0
iou_loss=ciou
nms_kind=diounms
beta_nms=0.6
new_coords=1
max_delta=2


# P4
[route]
layers = 77

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
#activation=linear
activation=logistic

[yolo]
mask = 3,4,5
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
classes={classes}
num=9
jitter=.1
scale_x_y = 2.0
objectness_smooth=1
ignore_thresh = .7
truth_thresh = 1
#random=1
resize=1.5
iou_thresh=0.2
iou_normalizer=0.05
cls_normalizer=0.5
obj_normalizer=1.0
iou_loss=ciou
nms_kind=diounms
beta_nms=0.6
new_coords=1
max_delta=2


# P5
[route]
layers = 86

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
#activation=linear
activation=logistic

[yolo]
mask = 6,7,8
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
classes={classes}
num=9
jitter=.1
scale_x_y = 2.0
objectness_smooth=1
ignore_thresh = .7
truth_thresh = 1
#random=1
resize=1.5
iou_thresh=0.2
iou_normalizer=0.05
cls_normalizer=0.5
obj_normalizer=1.0
iou_loss=ciou
nms_kind=diounms
beta_nms=0.6
new_coords=1
max_delta=2
)";