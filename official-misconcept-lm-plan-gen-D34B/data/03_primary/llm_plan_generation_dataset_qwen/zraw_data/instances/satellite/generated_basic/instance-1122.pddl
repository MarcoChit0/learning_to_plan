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
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	satellite5 - satellite
	instrument12 - instrument
	instrument13 - instrument
	thermograph0 - mode
	star3 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation1 - direction
	star5 - direction
	groundstation0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 star5)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation4)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 groundstation2)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star5)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation0)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star5)
)
(:goal (and
	(pointing satellite5 groundstation2)
	(have_image planet6 thermograph0)
))

)
