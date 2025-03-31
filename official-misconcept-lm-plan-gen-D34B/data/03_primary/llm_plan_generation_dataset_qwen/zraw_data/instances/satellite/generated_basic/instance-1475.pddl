(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	infrared0 - mode
	thermograph1 - mode
	star0 - direction
	groundstation2 - direction
	groundstation1 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument5 thermograph1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star3)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star3)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation2)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
	(supports instrument12 infrared0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star3)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star3)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
)
(:goal (and
	(pointing satellite4 star5)
	(have_image phenomenon4 infrared0)
	(have_image star5 thermograph1)
	(have_image star6 infrared0)
))

)
