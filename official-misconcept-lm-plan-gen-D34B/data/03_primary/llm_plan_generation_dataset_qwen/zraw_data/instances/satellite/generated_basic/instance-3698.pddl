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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	infrared2 - mode
	infrared0 - mode
	thermograph1 - mode
	star5 - direction
	groundstation2 - direction
	star4 - direction
	groundstation3 - direction
	star1 - direction
	star0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star4)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation2)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 star5)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star0)
	(calibration_target instrument7 star4)
	(supports instrument8 infrared0)
	(supports instrument8 infrared2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(supports instrument10 infrared2)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(calibration_target instrument10 groundstation3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star0)
	(calibration_target instrument11 star1)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image planet6 thermograph1)
))

)
