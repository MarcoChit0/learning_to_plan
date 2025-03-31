(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared0 - mode
	image1 - mode
	spectrograph2 - mode
	groundstation0 - direction
	star2 - direction
	star3 - direction
	groundstation1 - direction
	groundstation4 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(pointing satellite1 star7)
	(have_image phenomenon5 spectrograph2)
	(have_image star6 image1)
	(have_image star7 spectrograph2)
	(have_image star8 infrared0)
	(have_image phenomenon9 image1)
))

)
