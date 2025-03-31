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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	image0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation5 - direction
	groundstation1 - direction
	star6 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation3)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation4)
	(calibration_target instrument7 groundstation2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation5)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation3)
	(calibration_target instrument10 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 groundstation5)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 groundstation5)
	(pointing satellite1 groundstation1)
	(have_image star6 image0)
))

)
