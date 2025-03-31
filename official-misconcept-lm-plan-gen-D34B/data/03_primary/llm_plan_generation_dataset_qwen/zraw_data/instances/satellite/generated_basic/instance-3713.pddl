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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image0 - mode
	thermograph2 - mode
	thermograph1 - mode
	groundstation1 - direction
	groundstation3 - direction
	groundstation5 - direction
	star0 - direction
	star4 - direction
	star2 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation3)
	(calibration_target instrument6 groundstation5)
	(supports instrument7 image0)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star0)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 thermograph2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star4)
	(calibration_target instrument9 star0)
	(supports instrument10 image0)
	(calibration_target instrument10 star4)
	(calibration_target instrument10 star2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star2)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite2 star2)
	(have_image phenomenon6 thermograph1)
))

)
