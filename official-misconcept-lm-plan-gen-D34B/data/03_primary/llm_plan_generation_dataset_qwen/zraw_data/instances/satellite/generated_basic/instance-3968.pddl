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
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	thermograph0 - mode
	image1 - mode
	star0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 image1)
	(calibration_target instrument10 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
	(supports instrument11 image1)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 star0)
	(supports instrument14 image1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star0)
	(supports instrument15 image1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star0)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
)
(:goal (and
	(pointing satellite0 planet1)
	(pointing satellite2 planet1)
	(have_image planet1 thermograph0)
))

)
