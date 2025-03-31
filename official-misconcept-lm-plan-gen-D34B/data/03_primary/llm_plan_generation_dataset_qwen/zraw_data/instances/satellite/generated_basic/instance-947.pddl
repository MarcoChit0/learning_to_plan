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
	instrument4 - instrument
	instrument5 - instrument
	infrared0 - mode
	thermograph2 - mode
	spectrograph3 - mode
	thermograph1 - mode
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph3)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite2 star5)
	(have_image planet4 thermograph1)
	(have_image star5 thermograph2)
	(have_image star6 spectrograph3)
))

)
