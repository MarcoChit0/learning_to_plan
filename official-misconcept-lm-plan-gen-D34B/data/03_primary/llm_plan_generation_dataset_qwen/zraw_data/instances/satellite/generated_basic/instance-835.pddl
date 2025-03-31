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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	thermograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
	planet3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument9 infrared0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 thermograph1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 thermograph1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation0)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
	(supports instrument16 infrared0)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 infrared0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 thermograph1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 thermograph1)
	(calibration_target instrument20 groundstation0)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite3 groundstation0)
	(pointing satellite4 planet5)
	(have_image phenomenon1 thermograph1)
	(have_image phenomenon2 thermograph1)
	(have_image planet3 thermograph1)
	(have_image star4 infrared0)
	(have_image planet5 infrared0)
))

)
