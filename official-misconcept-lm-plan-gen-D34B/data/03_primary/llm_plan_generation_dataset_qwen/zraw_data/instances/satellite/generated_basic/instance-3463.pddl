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
	image1 - mode
	thermograph0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument4 image1)
	(calibration_target instrument4 star2)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star2)
	(pointing satellite2 phenomenon4)
	(have_image star3 image1)
	(have_image phenomenon4 thermograph0)
))

)
