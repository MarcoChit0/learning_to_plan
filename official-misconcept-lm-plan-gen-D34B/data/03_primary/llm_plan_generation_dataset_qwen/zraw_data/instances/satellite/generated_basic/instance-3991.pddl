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
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite2 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite3 - satellite
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	thermograph0 - mode
	image1 - mode
	groundstation1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(supports instrument7 thermograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star2)
	(supports instrument9 image1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(on_board instrument11 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image1)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph0)
	(supports instrument14 image1)
	(calibration_target instrument14 star2)
	(supports instrument15 image1)
	(calibration_target instrument15 star2)
	(supports instrument16 image1)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation0)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(on_board instrument16 satellite2)
	(on_board instrument17 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 star2)
	(supports instrument20 thermograph0)
	(supports instrument20 image1)
	(calibration_target instrument20 star2)
	(supports instrument21 image1)
	(supports instrument21 thermograph0)
	(calibration_target instrument21 groundstation0)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(on_board instrument20 satellite3)
	(on_board instrument21 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(have_image star3 image1)
))

)
