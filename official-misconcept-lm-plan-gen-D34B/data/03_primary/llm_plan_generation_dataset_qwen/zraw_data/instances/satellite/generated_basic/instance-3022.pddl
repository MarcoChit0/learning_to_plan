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
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	thermograph1 - mode
	image0 - mode
	star1 - direction
	star0 - direction
	groundstation3 - direction
	star2 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument8 thermograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star1)
	(supports instrument11 image0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star0)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument13 image0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 star2)
	(supports instrument14 image0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument16 image0)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 star2)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite2 phenomenon4)
	(pointing satellite3 star6)
	(pointing satellite4 groundstation3)
	(have_image phenomenon4 image0)
	(have_image star5 thermograph1)
	(have_image star6 thermograph1)
))

)
