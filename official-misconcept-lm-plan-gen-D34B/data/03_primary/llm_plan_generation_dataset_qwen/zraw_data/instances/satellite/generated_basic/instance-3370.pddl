(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	image1 - mode
	image2 - mode
	image3 - mode
	infrared0 - mode
	groundstation1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 image2)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image2)
	(supports instrument3 image3)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(supports instrument4 image2)
	(calibration_target instrument4 star0)
	(supports instrument5 image3)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image1)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 star0)
	(pointing satellite2 star0)
	(have_image star2 image1)
	(have_image star3 image2)
))

)
