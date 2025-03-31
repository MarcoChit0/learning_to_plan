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
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image1 - mode
	infrared2 - mode
	thermograph0 - mode
	groundstation2 - direction
	star3 - direction
	groundstation0 - direction
	groundstation1 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 star3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 thermograph0)
	(supports instrument7 infrared2)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared2)
	(supports instrument11 image1)
	(calibration_target instrument11 star3)
	(supports instrument12 image1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation0)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph0)
	(supports instrument15 infrared2)
	(supports instrument15 image1)
	(calibration_target instrument15 star3)
	(supports instrument16 infrared2)
	(calibration_target instrument16 groundstation0)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 image1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 groundstation0)
	(supports instrument19 image1)
	(calibration_target instrument19 groundstation1)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation2)
)
(:goal (and
	(pointing satellite3 groundstation2)
	(pointing satellite4 groundstation0)
	(have_image star4 infrared2)
	(have_image phenomenon5 image1)
))

)
