# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import pytest

import init_paths
from image_processing import imagecoor2cartesian_center

def test_imagecoor2cartesian_center():
	image_shape = (480, 640)
	forward, backward = imagecoor2cartesian_center(image_shape)
	
	test_pts = (0, 0)
	centered_pts = forward(test_pts)
	assert centered_pts == (-320, 240)
	back_pts = backward(centered_pts)
	assert back_pts == (0, 0)

	test_pts = (639, 479)
	centered_pts = forward(test_pts)
	assert centered_pts == (319, -239)
	back_pts = backward(centered_pts)
	assert back_pts == (639, 479)

	test_pts = (0, 479)
	centered_pts = forward(test_pts)
	assert centered_pts == (-320, -239)
	back_pts = backward(centered_pts)
	assert back_pts == (0, 479)

	test_pts = (639, 0)
	centered_pts = forward(test_pts)
	assert centered_pts == (319, 240)
	back_pts = backward(centered_pts)
	assert back_pts == (639, 0)


if __name__ == '__main__':
	pytest.main([__file__])