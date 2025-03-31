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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
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
	instrument23 - instrument
	instrument24 - instrument
	image0 - mode
	star1 - direction
	star3 - direction
	groundstation2 - direction
	groundstation5 - direction
	star4 - direction
	star0 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 image0)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(calibration_target instrument6 star4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument7 image0)
	(calibration_target instrument7 star3)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 star1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(calibration_target instrument9 star4)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 image0)
	(calibration_target instrument12 star4)
	(calibration_target instrument12 groundstation2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument13 image0)
	(calibration_target instrument13 star4)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 image0)
	(calibration_target instrument16 star3)
	(calibration_target instrument16 star0)
	(supports instrument17 image0)
	(calibration_target instrument17 star4)
	(supports instrument18 image0)
	(calibration_target instrument18 star0)
	(calibration_target instrument18 star4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation5)
	(supports instrument20 image0)
	(calibration_target instrument20 star4)
	(calibration_target instrument20 star3)
	(supports instrument21 image0)
	(calibration_target instrument21 groundstation2)
	(supports instrument22 image0)
	(calibration_target instrument22 star4)
	(supports instrument23 image0)
	(calibration_target instrument23 star4)
	(calibration_target instrument23 groundstation5)
	(supports instrument24 image0)
	(calibration_target instrument24 star0)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(on_board instrument23 satellite5)
	(on_board instrument24 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite1 groundstation5)
	(pointing satellite5 groundstation5)
	(have_image star6 image0)
))

)
