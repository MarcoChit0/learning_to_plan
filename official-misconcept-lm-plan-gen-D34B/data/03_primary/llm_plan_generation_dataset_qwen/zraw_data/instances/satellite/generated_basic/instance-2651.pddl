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
	satellite3 - satellite
	instrument6 - instrument
	image0 - mode
	groundstation1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star0 - direction
	star4 - direction
	star5 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 image0)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(supports instrument4 image0)
	(calibration_target instrument4 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 image0)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 star4)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image star6 image0)
))

)
