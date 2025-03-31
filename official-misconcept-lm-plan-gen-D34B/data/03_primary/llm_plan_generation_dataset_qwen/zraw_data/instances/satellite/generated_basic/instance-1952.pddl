(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	image0 - mode
	star1 - direction
	groundstation3 - direction
	star0 - direction
	groundstation2 - direction
	star4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(have_image star4 image0)
	(have_image phenomenon5 image0)
))

)
