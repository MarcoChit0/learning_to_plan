(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	image1 - mode
	thermograph0 - mode
	star1 - direction
	star4 - direction
	groundstation0 - direction
	groundstation2 - direction
	groundstation3 - direction
	star5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 image1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star4)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 star4)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 image1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 thermograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 groundstation3)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet7)
)
(:goal (and
	(have_image star5 image1)
	(have_image star6 thermograph0)
	(have_image planet7 image1)
	(have_image planet8 image1)
))

)
