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
	instrument6 - instrument
	image0 - mode
	image2 - mode
	image1 - mode
	star1 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
	star5 - direction
	star4 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image2)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(supports instrument3 image2)
	(calibration_target instrument3 star4)
	(supports instrument4 image1)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(supports instrument5 image0)
	(calibration_target instrument5 star3)
	(calibration_target instrument5 star5)
	(supports instrument6 image2)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 star5)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(have_image planet6 image0)
	(have_image phenomenon7 image2)
	(have_image planet8 image2)
	(have_image planet9 image2)
))

)
