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
	instrument11 - instrument
	satellite2 - satellite
	instrument12 - instrument
	instrument13 - instrument
	infrared1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation1 - direction
	star4 - direction
	star3 - direction
	groundstation5 - direction
	groundstation2 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument6 infrared1)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star4)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star4)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star3)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation5)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(on_board instrument11 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument12 infrared1)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 infrared1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(have_image phenomenon6 infrared0)
))

)
