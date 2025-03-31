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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	thermograph0 - mode
	thermograph1 - mode
	star1 - direction
	groundstation3 - direction
	star2 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star4)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star2)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star4)
	(supports instrument13 thermograph0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation0)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(pointing satellite0 star7)
	(pointing satellite3 groundstation0)
	(have_image star5 thermograph1)
	(have_image phenomenon6 thermograph0)
	(have_image star7 thermograph1)
	(have_image phenomenon8 thermograph0)
))

)
