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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite5 - satellite
	instrument20 - instrument
	instrument21 - instrument
	image0 - mode
	star0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 image0)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 image0)
	(calibration_target instrument14 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon1)
	(supports instrument15 image0)
	(calibration_target instrument15 star0)
	(supports instrument16 image0)
	(calibration_target instrument16 star0)
	(supports instrument17 image0)
	(calibration_target instrument17 star0)
	(supports instrument18 image0)
	(calibration_target instrument18 star0)
	(supports instrument19 image0)
	(calibration_target instrument19 star0)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon2)
	(supports instrument20 image0)
	(calibration_target instrument20 star0)
	(supports instrument21 image0)
	(calibration_target instrument21 star0)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon2)
)
(:goal (and
	(pointing satellite0 phenomenon2)
	(pointing satellite1 star0)
	(pointing satellite4 phenomenon2)
	(pointing satellite5 star0)
	(have_image phenomenon1 image0)
	(have_image phenomenon2 image0)
))

)
