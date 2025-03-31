(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	spectrograph3 - mode
	thermograph1 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph3)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(have_image planet2 thermograph1)
	(have_image planet3 spectrograph2)
))

)
