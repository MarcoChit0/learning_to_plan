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
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared2 - mode
	image0 - mode
	infrared3 - mode
	infrared1 - mode
	star0 - direction
	groundstation4 - direction
	groundstation1 - direction
	star2 - direction
	groundstation3 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(supports instrument2 infrared3)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image0)
	(supports instrument4 infrared3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 infrared2)
	(supports instrument5 image0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared2)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument7 image0)
	(supports instrument7 infrared1)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation1)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument8 infrared2)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared3)
	(calibration_target instrument9 star2)
	(supports instrument10 infrared1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 infrared2)
	(supports instrument11 image0)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star2)
	(supports instrument12 infrared3)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite1 groundstation1)
	(have_image planet5 infrared1)
	(have_image star6 infrared3)
	(have_image planet7 image0)
))

)
