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
	satellite2 - satellite
	instrument9 - instrument
	infrared1 - mode
	infrared0 - mode
	star0 - direction
	star4 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation3 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 groundstation3)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation3)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image planet5 infrared1)
))

)
