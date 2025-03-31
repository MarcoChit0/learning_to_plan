(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	image1 - mode
	thermograph0 - mode
	thermograph2 - mode
	infrared3 - mode
	star2 - direction
	star0 - direction
	star1 - direction
	planet3 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
	star7 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared3)
	(supports instrument4 thermograph2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared3)
	(calibration_target instrument6 star0)
	(supports instrument7 image1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared3)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 image1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
)
(:goal (and
	(have_image planet3 image1)
	(have_image star4 image1)
	(have_image star5 infrared3)
	(have_image planet6 infrared3)
	(have_image star7 thermograph0)
))

)
