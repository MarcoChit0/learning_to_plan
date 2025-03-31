(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite5 - satellite
	instrument20 - instrument
	image1 - mode
	image0 - mode
	star1 - direction
	groundstation0 - direction
	star2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 star1)
	(supports instrument9 image1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument11 image1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 image1)
	(supports instrument12 image0)
	(calibration_target instrument12 star1)
	(supports instrument13 image0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 image0)
	(supports instrument15 image1)
	(calibration_target instrument15 star1)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument16 image0)
	(calibration_target instrument16 star1)
	(supports instrument17 image0)
	(supports instrument17 image1)
	(calibration_target instrument17 star1)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 image1)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation0)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument20 image0)
	(supports instrument20 image1)
	(calibration_target instrument20 star2)
	(on_board instrument20 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star1)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 planet3)
	(pointing satellite4 groundstation0)
	(pointing satellite5 planet3)
	(have_image planet3 image0)
))

)
