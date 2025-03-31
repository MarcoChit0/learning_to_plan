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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite4 - satellite
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	infrared0 - mode
	star1 - direction
	star2 - direction
	star3 - direction
	star0 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star3)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(supports instrument14 infrared0)
	(calibration_target instrument14 star3)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument15 infrared0)
	(calibration_target instrument15 star0)
	(supports instrument16 infrared0)
	(calibration_target instrument16 star2)
	(supports instrument17 infrared0)
	(calibration_target instrument17 star1)
	(supports instrument18 infrared0)
	(calibration_target instrument18 star1)
	(supports instrument19 infrared0)
	(calibration_target instrument19 star0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument20 infrared0)
	(calibration_target instrument20 star1)
	(supports instrument21 infrared0)
	(calibration_target instrument21 star2)
	(supports instrument22 infrared0)
	(calibration_target instrument22 star3)
	(supports instrument23 infrared0)
	(calibration_target instrument23 star0)
	(on_board instrument20 satellite4)
	(on_board instrument21 satellite4)
	(on_board instrument22 satellite4)
	(on_board instrument23 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite0 phenomenon5)
	(pointing satellite1 star2)
	(have_image planet4 infrared0)
	(have_image phenomenon5 infrared0)
))

)
