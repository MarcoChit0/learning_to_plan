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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared0 - mode
	star3 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation0 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 infrared0)
	(calibration_target instrument14 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(have_image phenomenon5 infrared0)
	(have_image planet6 infrared0)
	(have_image star7 infrared0)
	(have_image phenomenon8 infrared0)
))

)
