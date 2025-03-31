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
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	thermograph1 - mode
	infrared0 - mode
	infrared3 - mode
	infrared2 - mode
	star1 - direction
	star0 - direction
	phenomenon2 - direction
	star3 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared0)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared0)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star1)
	(supports instrument7 thermograph1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared2)
	(calibration_target instrument9 star0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star1)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph1)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph1)
	(supports instrument12 infrared2)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared2)
	(supports instrument13 infrared3)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star0)
	(supports instrument14 thermograph1)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph1)
	(supports instrument15 infrared0)
	(supports instrument15 infrared2)
	(calibration_target instrument15 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument16 infrared2)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 star1)
	(supports instrument17 thermograph1)
	(supports instrument17 infrared3)
	(calibration_target instrument17 star0)
	(supports instrument18 infrared3)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite2 star3)
	(pointing satellite3 star1)
	(have_image phenomenon2 thermograph1)
	(have_image star3 infrared0)
))

)
