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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	thermograph1 - mode
	infrared0 - mode
	infrared2 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 infrared2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star1)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared2)
	(calibration_target instrument11 star1)
	(supports instrument12 infrared0)
	(supports instrument12 infrared2)
	(calibration_target instrument12 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument13 infrared0)
	(supports instrument13 infrared2)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation2)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
	(supports instrument15 thermograph1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared0)
	(calibration_target instrument16 groundstation2)
	(supports instrument17 thermograph1)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star1)
	(supports instrument18 thermograph1)
	(supports instrument18 infrared2)
	(calibration_target instrument18 groundstation2)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite2 groundstation2)
	(have_image planet3 infrared0)
))

)
