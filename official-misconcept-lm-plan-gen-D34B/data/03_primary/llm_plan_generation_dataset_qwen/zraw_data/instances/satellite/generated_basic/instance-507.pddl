(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image0 - mode
	image2 - mode
	infrared1 - mode
	spectrograph3 - mode
	groundstation2 - direction
	groundstation4 - direction
	star0 - direction
	star1 - direction
	star3 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 image2)
	(supports instrument2 spectrograph3)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(have_image star5 image0)
	(have_image star6 image2)
	(have_image star7 spectrograph3)
	(have_image phenomenon8 image2)
))

)
