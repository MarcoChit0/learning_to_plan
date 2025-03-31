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
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	infrared1 - mode
	image0 - mode
	star0 - direction
	planet1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
	(supports instrument10 image0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star0)
	(supports instrument11 infrared1)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 infrared1)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
	(supports instrument13 infrared1)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet1 image0)
	(have_image planet2 image0)
))

)
