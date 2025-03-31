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
	satellite2 - satellite
	instrument5 - instrument
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
	satellite5 - satellite
	instrument18 - instrument
	thermograph1 - mode
	infrared2 - mode
	image0 - mode
	star1 - direction
	star3 - direction
	groundstation2 - direction
	groundstation0 - direction
	star4 - direction
	star5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image0)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared2)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 image0)
	(supports instrument5 infrared2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared2)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image0)
	(supports instrument7 infrared2)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 thermograph1)
	(supports instrument8 infrared2)
	(supports instrument8 image0)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared2)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph1)
	(supports instrument10 image0)
	(supports instrument10 infrared2)
	(calibration_target instrument10 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument11 infrared2)
	(calibration_target instrument11 star3)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 image0)
	(supports instrument13 infrared2)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation2)
	(supports instrument14 infrared2)
	(supports instrument14 image0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star1)
	(supports instrument15 infrared2)
	(supports instrument15 image0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument16 thermograph1)
	(supports instrument16 infrared2)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 image0)
	(supports instrument17 infrared2)
	(calibration_target instrument17 groundstation2)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation0)
	(supports instrument18 infrared2)
	(supports instrument18 thermograph1)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation0)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation0)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 star1)
	(pointing satellite3 star3)
	(pointing satellite4 groundstation2)
	(pointing satellite5 star1)
	(have_image star4 thermograph1)
	(have_image star5 infrared2)
))

)
