(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image1 - mode
	image0 - mode
	spectrograph2 - mode
	star0 - direction
	star2 - direction
	star4 - direction
	groundstation3 - direction
	star1 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(have_image phenomenon5 image1)
))

)
