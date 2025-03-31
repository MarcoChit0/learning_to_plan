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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image2 - mode
	thermograph0 - mode
	infrared1 - mode
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 image2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph0)
	(supports instrument6 image2)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 image2)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph0)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image2)
	(supports instrument11 infrared1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument13 infrared1)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image2)
	(calibration_target instrument15 star2)
	(supports instrument16 infrared1)
	(supports instrument16 thermograph0)
	(supports instrument16 image2)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 image2)
	(supports instrument18 thermograph0)
	(supports instrument18 infrared1)
	(calibration_target instrument18 groundstation1)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
)
(:goal (and
	(pointing satellite3 groundstation0)
	(have_image phenomenon3 infrared1)
))

)
