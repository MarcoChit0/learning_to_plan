(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	spectrograph2 - mode
	spectrograph0 - mode
	image1 - mode
	thermograph3 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 image1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph3)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(have_image star3 image1)
	(have_image phenomenon4 image1)
	(have_image phenomenon5 image1)
))

)
