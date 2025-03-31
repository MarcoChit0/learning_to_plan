(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
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
	image0 - mode
	star1 - direction
	star3 - direction
	star4 - direction
	groundstation0 - direction
	groundstation5 - direction
	star2 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 star4)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 star4)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument9 image0)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 star4)
	(supports instrument10 image0)
	(calibration_target instrument10 star3)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation0)
	(calibration_target instrument12 groundstation5)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation5)
	(calibration_target instrument13 star4)
	(supports instrument14 image0)
	(calibration_target instrument14 star2)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation5)
	(calibration_target instrument15 star2)
	(supports instrument16 image0)
	(calibration_target instrument16 star3)
	(supports instrument17 image0)
	(calibration_target instrument17 star4)
	(supports instrument18 image0)
	(calibration_target instrument18 star4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation0)
	(calibration_target instrument19 star2)
	(supports instrument20 image0)
	(calibration_target instrument20 star2)
	(calibration_target instrument20 groundstation5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation5)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(pointing satellite1 star4)
	(have_image phenomenon6 image0)
))

)
