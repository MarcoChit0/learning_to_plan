(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	image0 - mode
	star4 - direction
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star4)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 star4)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation3)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image0)
	(calibration_target instrument14 star2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite1 groundstation1)
	(have_image phenomenon5 image0)
	(have_image planet6 image0)
	(have_image phenomenon7 image0)
	(have_image phenomenon8 image0)
))

)
