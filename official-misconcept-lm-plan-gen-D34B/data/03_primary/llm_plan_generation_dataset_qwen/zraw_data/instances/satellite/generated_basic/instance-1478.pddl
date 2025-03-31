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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite4 - satellite
	instrument17 - instrument
	image1 - mode
	thermograph0 - mode
	groundstation0 - direction
	star3 - direction
	star2 - direction
	groundstation1 - direction
	planet4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image1)
	(calibration_target instrument8 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 star2)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 star3)
	(supports instrument11 image1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 thermograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 star3)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument13 image1)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph0)
	(supports instrument14 image1)
	(calibration_target instrument14 star2)
	(supports instrument15 image1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star3)
	(supports instrument16 thermograph0)
	(supports instrument16 image1)
	(calibration_target instrument16 star2)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument17 thermograph0)
	(supports instrument17 image1)
	(calibration_target instrument17 groundstation1)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 groundstation1)
	(pointing satellite3 phenomenon6)
	(pointing satellite4 phenomenon5)
	(have_image planet4 thermograph0)
	(have_image phenomenon5 image1)
	(have_image phenomenon6 thermograph0)
))

)
