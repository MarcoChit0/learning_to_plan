(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	spectrograph2 - mode
	image0 - mode
	thermograph3 - mode
	thermograph1 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 thermograph3)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph3)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(have_image phenomenon4 spectrograph2)
))

)
