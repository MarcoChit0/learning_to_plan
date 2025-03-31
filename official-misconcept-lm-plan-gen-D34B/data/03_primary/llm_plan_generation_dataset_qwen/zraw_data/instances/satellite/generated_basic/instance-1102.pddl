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
	instrument15 - instrument
	satellite3 - satellite
	instrument16 - instrument
	instrument17 - instrument
	satellite4 - satellite
	instrument18 - instrument
	instrument19 - instrument
	satellite5 - satellite
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	instrument23 - instrument
	instrument24 - instrument
	instrument25 - instrument
	image0 - mode
	star1 - direction
	star5 - direction
	star4 - direction
	star3 - direction
	star0 - direction
	star2 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument5 image0)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 star1)
	(supports instrument6 image0)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 star5)
	(supports instrument7 image0)
	(calibration_target instrument7 star5)
	(supports instrument8 image0)
	(calibration_target instrument8 star3)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 star2)
	(supports instrument10 image0)
	(calibration_target instrument10 star1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument11 image0)
	(calibration_target instrument11 star2)
	(supports instrument12 image0)
	(calibration_target instrument12 star2)
	(supports instrument13 image0)
	(calibration_target instrument13 star4)
	(calibration_target instrument13 star0)
	(supports instrument14 image0)
	(calibration_target instrument14 star1)
	(supports instrument15 image0)
	(calibration_target instrument15 star5)
	(calibration_target instrument15 star4)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument16 image0)
	(calibration_target instrument16 star0)
	(calibration_target instrument16 star5)
	(supports instrument17 image0)
	(calibration_target instrument17 star4)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument18 image0)
	(calibration_target instrument18 star5)
	(supports instrument19 image0)
	(calibration_target instrument19 star2)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument20 image0)
	(calibration_target instrument20 star0)
	(calibration_target instrument20 star4)
	(supports instrument21 image0)
	(calibration_target instrument21 star3)
	(supports instrument22 image0)
	(calibration_target instrument22 star4)
	(calibration_target instrument22 star2)
	(supports instrument23 image0)
	(calibration_target instrument23 star2)
	(calibration_target instrument23 star4)
	(supports instrument24 image0)
	(calibration_target instrument24 star0)
	(calibration_target instrument24 star3)
	(supports instrument25 image0)
	(calibration_target instrument25 star2)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(on_board instrument25 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star4)
)
(:goal (and
	(pointing satellite4 star3)
	(pointing satellite5 star3)
	(have_image star6 image0)
))

)
