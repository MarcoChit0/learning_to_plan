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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	infrared0 - mode
	infrared3 - mode
	image1 - mode
	infrared2 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared3)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument4 infrared3)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared2)
	(supports instrument5 infrared0)
	(supports instrument5 infrared3)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument7 infrared0)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star0)
	(supports instrument8 infrared3)
	(supports instrument8 image1)
	(calibration_target instrument8 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 phenomenon2)
	(have_image phenomenon2 infrared0)
	(have_image phenomenon3 image1)
))

)
