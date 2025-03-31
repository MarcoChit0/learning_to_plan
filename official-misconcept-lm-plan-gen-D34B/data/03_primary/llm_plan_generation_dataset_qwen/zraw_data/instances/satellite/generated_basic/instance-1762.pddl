(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	image0 - mode
	thermograph1 - mode
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	star3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument13 thermograph1)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star2)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 thermograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation1)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite1 star2)
	(have_image star3 thermograph1)
))

)
