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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	image1 - mode
	image2 - mode
	thermograph0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	planet3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 image2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star1)
	(supports instrument3 image2)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 image2)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star1)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image star2 thermograph0)
	(have_image planet3 thermograph0)
	(have_image planet4 thermograph0)
))

)
