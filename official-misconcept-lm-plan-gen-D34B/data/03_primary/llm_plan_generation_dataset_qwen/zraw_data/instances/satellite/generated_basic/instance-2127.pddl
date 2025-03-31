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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	instrument22 - instrument
	image1 - mode
	thermograph0 - mode
	groundstation0 - direction
	groundstation1 - direction
	planet2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation0)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument12 thermograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 thermograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument14 image1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 thermograph0)
	(supports instrument16 image1)
	(calibration_target instrument16 groundstation1)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation1)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 thermograph0)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 image1)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 groundstation0)
	(supports instrument20 image1)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 groundstation0)
	(supports instrument21 image1)
	(calibration_target instrument21 groundstation0)
	(supports instrument22 thermograph0)
	(calibration_target instrument22 groundstation1)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(on_board instrument22 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation0)
	(pointing satellite3 phenomenon4)
	(pointing satellite4 star3)
	(have_image planet2 thermograph0)
	(have_image star3 thermograph0)
	(have_image phenomenon4 image1)
))

)
