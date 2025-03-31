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
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite5 - satellite
	instrument14 - instrument
	instrument15 - instrument
	image0 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation1)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
	(supports instrument14 image0)
	(calibration_target instrument14 star0)
	(supports instrument15 image0)
	(calibration_target instrument15 groundstation1)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite3 star0)
	(have_image star2 image0)
	(have_image star3 image0)
	(have_image planet4 image0)
))

)
