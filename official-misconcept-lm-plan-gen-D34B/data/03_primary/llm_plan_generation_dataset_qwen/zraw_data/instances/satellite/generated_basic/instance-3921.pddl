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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	infrared1 - mode
	thermograph0 - mode
	star0 - direction
	star5 - direction
	groundstation1 - direction
	star2 - direction
	groundstation4 - direction
	groundstation3 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 star5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 star5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation3)
	(calibration_target instrument6 star5)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation4)
	(calibration_target instrument8 star5)
	(supports instrument9 thermograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation4)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument11 infrared1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument12 infrared1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation4)
	(calibration_target instrument12 star2)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
	(supports instrument13 thermograph0)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation3)
	(on_board instrument13 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(have_image planet6 thermograph0)
))

)
