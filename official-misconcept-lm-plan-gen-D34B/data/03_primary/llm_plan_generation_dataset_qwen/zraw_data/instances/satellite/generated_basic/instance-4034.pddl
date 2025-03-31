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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared1 - mode
	thermograph2 - mode
	thermograph0 - mode
	star0 - direction
	star1 - direction
	star2 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph0)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(have_image star2 thermograph2)
))

)
