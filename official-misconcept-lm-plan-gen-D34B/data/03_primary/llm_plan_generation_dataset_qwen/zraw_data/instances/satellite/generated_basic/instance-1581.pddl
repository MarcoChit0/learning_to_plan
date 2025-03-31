(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	satellite5 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image0 - mode
	groundstation3 - direction
	groundstation2 - direction
	star4 - direction
	star0 - direction
	star1 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star4)
	(supports instrument1 image0)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image0)
	(calibration_target instrument10 star4)
	(supports instrument11 image0)
	(calibration_target instrument11 star1)
	(supports instrument12 image0)
	(calibration_target instrument12 star1)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation3)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 image0)
	(calibration_target instrument16 star4)
	(supports instrument17 image0)
	(calibration_target instrument17 star0)
	(supports instrument18 image0)
	(calibration_target instrument18 star0)
	(supports instrument19 image0)
	(calibration_target instrument19 star1)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation2)
)
(:goal (and
	(pointing satellite0 planet5)
	(pointing satellite3 groundstation3)
	(have_image planet5 image0)
	(have_image phenomenon6 image0)
	(have_image planet7 image0)
))

)
