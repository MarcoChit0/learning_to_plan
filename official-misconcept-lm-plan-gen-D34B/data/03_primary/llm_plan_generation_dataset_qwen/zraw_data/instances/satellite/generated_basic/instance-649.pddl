(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	thermograph0 - mode
	thermograph1 - mode
	image2 - mode
	star2 - direction
	star3 - direction
	groundstation0 - direction
	star1 - direction
	star4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 star3)
	(supports instrument1 image2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 image2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 image2)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star3)
	(supports instrument6 image2)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star3)
	(supports instrument7 image2)
	(supports instrument7 thermograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 thermograph1)
	(supports instrument8 image2)
	(calibration_target instrument8 star1)
	(supports instrument9 image2)
	(calibration_target instrument9 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument10 image2)
	(supports instrument10 thermograph0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star3)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 image2)
	(supports instrument14 thermograph0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star1)
	(supports instrument15 image2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star3)
	(supports instrument16 image2)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 image2)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation0)
	(supports instrument18 image2)
	(supports instrument18 thermograph1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 star1)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(have_image star4 thermograph1)
	(have_image phenomenon5 thermograph1)
	(have_image star6 thermograph1)
))

)
