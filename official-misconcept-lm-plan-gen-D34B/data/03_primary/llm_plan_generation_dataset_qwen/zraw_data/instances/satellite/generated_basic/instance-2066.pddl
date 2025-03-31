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
	satellite3 - satellite
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite5 - satellite
	instrument13 - instrument
	image0 - mode
	groundstation3 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	star4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument6 image0)
	(calibration_target instrument6 star2)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 image0)
	(calibration_target instrument11 star2)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star4)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation1)
	(on_board instrument13 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation1)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 groundstation1)
	(pointing satellite5 groundstation3)
	(have_image star4 image0)
))

)
