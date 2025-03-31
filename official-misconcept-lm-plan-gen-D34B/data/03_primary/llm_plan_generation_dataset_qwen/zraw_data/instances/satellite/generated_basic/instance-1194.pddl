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
	image2 - mode
	image3 - mode
	star1 - direction
	star0 - direction
	groundstation2 - direction
	star3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 image3)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image3)
	(supports instrument2 image2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image3)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image3)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image star3 image1)
	(have_image phenomenon4 image1)
	(have_image planet5 image1)
))

)
