(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	infrared0 - mode
	spectrograph1 - mode
	thermograph2 - mode
	groundstation1 - direction
	star4 - direction
	groundstation2 - direction
	star3 - direction
	star5 - direction
	star0 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star9)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(pointing satellite2 phenomenon8)
	(have_image star6 infrared0)
	(have_image star7 infrared0)
	(have_image phenomenon8 spectrograph1)
	(have_image star9 spectrograph1)
))

)
