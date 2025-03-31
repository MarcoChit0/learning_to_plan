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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	thermograph1 - mode
	thermograph2 - mode
	infrared3 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 thermograph1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph2)
	(supports instrument2 infrared3)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument6 image0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 star1)
	(supports instrument10 infrared3)
	(supports instrument10 image0)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared3)
	(supports instrument11 image0)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph1)
	(supports instrument13 thermograph2)
	(supports instrument13 infrared3)
	(calibration_target instrument13 star0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument14 thermograph1)
	(supports instrument14 thermograph2)
	(supports instrument14 image0)
	(calibration_target instrument14 star1)
	(supports instrument15 infrared3)
	(supports instrument15 thermograph2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star1)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 star0)
	(supports instrument17 infrared3)
	(supports instrument17 thermograph1)
	(supports instrument17 image0)
	(calibration_target instrument17 star1)
	(supports instrument18 image0)
	(supports instrument18 thermograph2)
	(calibration_target instrument18 star1)
	(supports instrument19 infrared3)
	(supports instrument19 image0)
	(supports instrument19 thermograph1)
	(calibration_target instrument19 star1)
	(supports instrument20 infrared3)
	(supports instrument20 thermograph2)
	(calibration_target instrument20 star1)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite3 star0)
	(have_image phenomenon2 image0)
	(have_image phenomenon3 thermograph1)
))

)
