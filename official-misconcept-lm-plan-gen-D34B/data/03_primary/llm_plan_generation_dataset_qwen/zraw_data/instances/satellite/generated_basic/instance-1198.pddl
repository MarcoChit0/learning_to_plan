(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	thermograph2 - mode
	thermograph0 - mode
	spectrograph1 - mode
	image3 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	planet3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 image3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image3)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 star2)
	(have_image planet3 thermograph0)
	(have_image star4 image3)
	(have_image planet5 thermograph0)
))

)
