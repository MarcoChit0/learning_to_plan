(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	image1 - mode
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star4)
	(supports instrument10 image1)
	(calibration_target instrument10 star4)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(supports instrument12 thermograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 star4)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite1 phenomenon5)
	(have_image phenomenon5 thermograph0)
))

)
