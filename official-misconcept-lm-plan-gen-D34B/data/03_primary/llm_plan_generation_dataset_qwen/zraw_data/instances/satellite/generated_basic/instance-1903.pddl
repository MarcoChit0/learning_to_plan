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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	image0 - mode
	star1 - direction
	star3 - direction
	star4 - direction
	groundstation2 - direction
	star0 - direction
	planet5 - direction
	phenomenon6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument3 image0)
	(calibration_target instrument3 star4)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 star3)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image0)
	(calibration_target instrument10 star4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument11 image0)
	(calibration_target instrument11 star4)
	(supports instrument12 image0)
	(calibration_target instrument12 star4)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 image0)
	(calibration_target instrument15 star0)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(pointing satellite1 star0)
	(pointing satellite2 phenomenon6)
	(have_image planet5 image0)
	(have_image phenomenon6 image0)
	(have_image star7 image0)
	(have_image phenomenon8 image0)
))

)
