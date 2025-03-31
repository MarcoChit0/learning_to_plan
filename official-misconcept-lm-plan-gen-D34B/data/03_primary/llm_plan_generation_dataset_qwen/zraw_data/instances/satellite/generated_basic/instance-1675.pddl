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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	star5 - direction
	star2 - direction
	star1 - direction
	star0 - direction
	groundstation4 - direction
	groundstation3 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 star5)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star5)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star2)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation4)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation3)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation3)
	(calibration_target instrument12 groundstation4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon6)
)
(:goal (and
	(pointing satellite3 star2)
	(have_image phenomenon6 thermograph0)
))

)
