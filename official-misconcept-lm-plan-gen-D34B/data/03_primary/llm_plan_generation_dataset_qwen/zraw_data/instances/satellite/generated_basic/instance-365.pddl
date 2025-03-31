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
	instrument6 - instrument
	thermograph0 - mode
	image1 - mode
	star2 - direction
	star1 - direction
	star0 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image1)
	(calibration_target instrument1 star1)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image phenomenon4 thermograph0)
	(have_image star5 image1)
	(have_image planet6 image1)
))

)
