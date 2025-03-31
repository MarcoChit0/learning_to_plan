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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	infrared1 - mode
	infrared0 - mode
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 infrared1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation3)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 infrared0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 infrared0)
	(supports instrument7 infrared1)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument8 infrared1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star0)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(pointing satellite4 phenomenon5)
	(have_image phenomenon4 infrared1)
	(have_image phenomenon5 infrared0)
	(have_image phenomenon6 infrared0)
))

)
