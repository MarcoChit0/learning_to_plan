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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared2 - mode
	infrared0 - mode
	image1 - mode
	star4 - direction
	star2 - direction
	groundstation5 - direction
	groundstation3 - direction
	groundstation0 - direction
	groundstation1 - direction
	planet6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star4)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 image1)
	(supports instrument3 infrared2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star4)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation5)
	(calibration_target instrument4 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star2)
	(supports instrument6 image1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 star2)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 infrared2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation5)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 infrared2)
	(supports instrument10 image1)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 image1)
	(supports instrument11 infrared2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
)
(:goal (and
	(pointing satellite2 groundstation0)
	(have_image planet6 infrared0)
	(have_image star7 infrared2)
	(have_image phenomenon8 infrared0)
	(have_image star9 image1)
))

)
