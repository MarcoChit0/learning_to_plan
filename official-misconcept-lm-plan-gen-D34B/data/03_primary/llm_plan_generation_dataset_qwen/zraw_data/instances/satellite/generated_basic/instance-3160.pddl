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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	satellite5 - satellite
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	thermograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument9 thermograph1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star2)
	(supports instrument12 thermograph1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation1)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet3)
	(supports instrument14 infrared0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 infrared0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 thermograph1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 infrared0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 groundstation1)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
	(supports instrument19 thermograph1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 star2)
	(supports instrument20 thermograph1)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star2)
	(supports instrument21 thermograph1)
	(calibration_target instrument21 star2)
	(supports instrument22 thermograph1)
	(calibration_target instrument22 star2)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite5 star5)
	(have_image planet3 infrared0)
	(have_image phenomenon4 infrared0)
	(have_image star5 thermograph1)
))

)
