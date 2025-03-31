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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	thermograph2 - mode
	thermograph1 - mode
	star2 - direction
	groundstation4 - direction
	groundstation3 - direction
	star5 - direction
	groundstation0 - direction
	groundstation1 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 star5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star5)
	(supports instrument10 thermograph2)
	(supports instrument10 thermograph1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation3)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star5)
	(supports instrument12 thermograph0)
	(supports instrument12 thermograph2)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation1)
	(calibration_target instrument12 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite1 planet6)
	(have_image planet6 thermograph1)
))

)
