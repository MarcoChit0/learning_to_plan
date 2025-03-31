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
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	star0 - direction
	star2 - direction
	star1 - direction
	groundstation4 - direction
	star3 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star3)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation4)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet8)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(have_image planet5 thermograph0)
	(have_image star6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image planet8 thermograph0)
))

)
