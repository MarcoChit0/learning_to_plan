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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	thermograph1 - mode
	infrared0 - mode
	star1 - direction
	groundstation5 - direction
	star2 - direction
	star4 - direction
	groundstation0 - direction
	star3 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star4)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument10 infrared0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation0)
	(calibration_target instrument10 star4)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation5)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(calibration_target instrument13 star4)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon8)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation5)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet7)
	(supports instrument16 thermograph1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation5)
	(calibration_target instrument16 star3)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 star4)
	(calibration_target instrument17 star2)
	(supports instrument18 infrared0)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 star3)
	(supports instrument19 infrared0)
	(supports instrument19 thermograph1)
	(calibration_target instrument19 groundstation0)
	(calibration_target instrument19 star3)
	(supports instrument20 thermograph1)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star3)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite1 star3)
	(pointing satellite3 planet7)
	(pointing satellite4 star3)
	(pointing satellite5 star2)
	(have_image planet6 thermograph1)
	(have_image planet7 infrared0)
	(have_image phenomenon8 infrared0)
))

)
