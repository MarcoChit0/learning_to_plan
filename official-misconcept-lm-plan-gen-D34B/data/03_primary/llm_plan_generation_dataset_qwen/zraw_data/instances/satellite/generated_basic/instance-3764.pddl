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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	image2 - mode
	thermograph1 - mode
	infrared0 - mode
	star2 - direction
	groundstation1 - direction
	star0 - direction
	star3 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star2)
	(supports instrument4 image2)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 image2)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument7 thermograph1)
	(supports instrument7 image2)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared0)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared0)
	(supports instrument9 image2)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star2)
	(supports instrument10 image2)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument12 image2)
	(supports instrument12 infrared0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph1)
	(supports instrument14 image2)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star2)
	(supports instrument15 image2)
	(supports instrument15 thermograph1)
	(supports instrument15 infrared0)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument16 thermograph1)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star2)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument17 image2)
	(supports instrument17 infrared0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 thermograph1)
	(supports instrument18 infrared0)
	(supports instrument18 image2)
	(calibration_target instrument18 star0)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star2)
)
(:goal (and
	(pointing satellite1 star2)
	(have_image star3 image2)
))

)
