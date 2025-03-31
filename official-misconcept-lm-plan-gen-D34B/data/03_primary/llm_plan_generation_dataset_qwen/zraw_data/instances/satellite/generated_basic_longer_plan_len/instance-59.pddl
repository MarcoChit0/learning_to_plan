(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	image0 - mode
	spectrograph1 - mode
	thermograph2 - mode
	star2 - direction
	star0 - direction
	groundstation1 - direction
	planet3 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 phenomenon4)
	(pointing satellite2 star6)
	(have_image planet3 image0)
	(have_image phenomenon4 thermograph2)
	(have_image star5 thermograph2)
	(have_image star6 spectrograph1)
	(have_image planet7 thermograph2)
))

)
