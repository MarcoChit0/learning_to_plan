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
	image0 - mode
	image1 - mode
	star1 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 image1)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image phenomenon4 image0)
	(have_image star5 image0)
	(have_image star6 image1)
))

)
