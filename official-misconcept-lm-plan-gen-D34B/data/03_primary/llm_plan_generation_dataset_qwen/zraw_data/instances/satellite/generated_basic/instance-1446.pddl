(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	spectrograph3 - mode
	infrared1 - mode
	spectrograph0 - mode
	thermograph2 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument3 spectrograph3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite2 star0)
	(have_image star1 infrared1)
	(have_image planet2 thermograph2)
	(have_image planet3 spectrograph0)
))

)
