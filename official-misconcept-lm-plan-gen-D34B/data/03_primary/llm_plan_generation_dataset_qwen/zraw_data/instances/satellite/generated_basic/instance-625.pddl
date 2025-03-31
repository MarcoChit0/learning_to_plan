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
	satellite3 - satellite
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared1 - mode
	thermograph2 - mode
	infrared0 - mode
	star2 - direction
	star1 - direction
	star0 - direction
	star3 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph2)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph2)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star3)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 star0)
	(pointing satellite3 star0)
	(have_image star4 thermograph2)
	(have_image star5 infrared0)
	(have_image planet6 infrared1)
))

)
