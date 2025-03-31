(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	infrared0 - mode
	spectrograph2 - mode
	thermograph1 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star9)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation3)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star9)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(have_image star4 infrared0)
	(have_image planet5 infrared0)
	(have_image star6 thermograph1)
	(have_image planet7 spectrograph2)
	(have_image planet8 infrared0)
	(have_image star9 spectrograph2)
))

)
