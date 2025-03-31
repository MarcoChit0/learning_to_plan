(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	thermograph0 - mode
	star1 - direction
	star2 - direction
	groundstation3 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star4)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star4)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star1)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star9)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star4)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star2)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation3)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star4)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 groundstation0)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star9)
)
(:goal (and
	(pointing satellite2 star4)
	(pointing satellite3 star2)
	(pointing satellite4 groundstation3)
	(pointing satellite5 star4)
	(have_image star5 thermograph0)
	(have_image star6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 thermograph0)
	(have_image star9 thermograph0)
	(have_image phenomenon10 thermograph0)
))

)
