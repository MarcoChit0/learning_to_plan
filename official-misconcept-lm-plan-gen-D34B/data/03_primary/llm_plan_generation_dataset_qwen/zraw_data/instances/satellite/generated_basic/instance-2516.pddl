(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	image0 - mode
	groundstation1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(calibration_target instrument6 star3)
	(supports instrument7 image0)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(supports instrument10 image0)
	(calibration_target instrument10 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument11 image0)
	(calibration_target instrument11 star3)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(supports instrument13 image0)
	(calibration_target instrument13 star3)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument14 image0)
	(calibration_target instrument14 star2)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument15 image0)
	(calibration_target instrument15 star3)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 groundstation1)
	(pointing satellite4 star5)
	(have_image planet4 image0)
	(have_image star5 image0)
))

)
