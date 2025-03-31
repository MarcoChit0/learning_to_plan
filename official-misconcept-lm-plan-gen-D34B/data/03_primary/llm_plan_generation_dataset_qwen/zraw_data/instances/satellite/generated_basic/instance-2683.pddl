(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image1 - mode
	image2 - mode
	spectrograph3 - mode
	infrared0 - mode
	star1 - direction
	groundstation3 - direction
	star0 - direction
	groundstation2 - direction
	star4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image2)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph3)
	(supports instrument2 image1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image star4 image1)
))

)
