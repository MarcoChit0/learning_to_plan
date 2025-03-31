(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	thermograph1 - mode
	image0 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star3)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument13 image0)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite0 star4)
	(pointing satellite1 star2)
	(have_image star4 image0)
))

)
