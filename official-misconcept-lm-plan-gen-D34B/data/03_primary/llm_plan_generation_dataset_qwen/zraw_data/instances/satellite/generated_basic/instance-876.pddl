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
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	thermograph0 - mode
	image1 - mode
	infrared2 - mode
	star0 - direction
	groundstation1 - direction
	star3 - direction
	star2 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon5)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared2)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared2)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 star3)
	(supports instrument12 thermograph0)
	(supports instrument12 infrared2)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument13 thermograph0)
	(supports instrument13 infrared2)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 infrared2)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 infrared2)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star2)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon5)
	(supports instrument16 image1)
	(calibration_target instrument16 star2)
	(supports instrument17 image1)
	(supports instrument17 infrared2)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 image1)
	(supports instrument18 infrared2)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 image1)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 groundstation1)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 star3)
	(supports instrument21 infrared2)
	(calibration_target instrument21 star2)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon5)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(pointing satellite3 star3)
	(pointing satellite5 star0)
	(have_image phenomenon4 infrared2)
	(have_image phenomenon5 infrared2)
))

)
