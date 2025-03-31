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
	image0 - mode
	star3 - direction
	star2 - direction
	groundstation0 - direction
	groundstation1 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 groundstation0)
	(have_image star4 image0)
	(have_image phenomenon5 image0)
	(have_image planet6 image0)
))

)
