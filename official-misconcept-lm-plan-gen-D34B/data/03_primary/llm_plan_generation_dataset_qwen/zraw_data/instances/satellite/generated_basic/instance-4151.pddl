(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	thermograph2 - mode
	thermograph1 - mode
	infrared3 - mode
	infrared0 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared3)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph1)
	(supports instrument8 thermograph2)
	(supports instrument8 infrared3)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph1)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared3)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star1)
	(supports instrument12 infrared0)
	(supports instrument12 infrared3)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph1)
	(supports instrument13 infrared3)
	(calibration_target instrument13 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
)
(:goal (and
	(pointing satellite0 planet2)
	(pointing satellite1 planet2)
	(pointing satellite2 planet2)
	(have_image planet2 infrared0)
))

)
