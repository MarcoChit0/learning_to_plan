(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph3 - mode
	thermograph0 - mode
	image1 - mode
	spectrograph2 - mode
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	star2 - direction
	star4 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 spectrograph2)
	(supports instrument1 thermograph3)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
)
(:goal (and
	(pointing satellite0 phenomenon7)
	(have_image planet5 thermograph3)
	(have_image phenomenon6 thermograph3)
	(have_image phenomenon7 image1)
	(have_image phenomenon8 thermograph3)
))

)
