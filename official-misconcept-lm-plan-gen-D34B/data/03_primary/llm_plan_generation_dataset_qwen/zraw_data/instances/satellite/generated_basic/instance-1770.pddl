(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph1 - mode
	image0 - mode
	groundstation2 - direction
	star1 - direction
	star0 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star1)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument13 thermograph1)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 image0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star1)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star0)
	(supports instrument16 thermograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 star0)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite0 phenomenon3)
	(pointing satellite4 groundstation2)
	(pointing satellite5 star1)
	(have_image phenomenon3 image0)
))

)
