(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image1 - mode
	image0 - mode
	infrared2 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 image1)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 star0)
	(have_image star2 image0)
	(have_image planet3 infrared2)
	(have_image phenomenon4 infrared2)
))

)
