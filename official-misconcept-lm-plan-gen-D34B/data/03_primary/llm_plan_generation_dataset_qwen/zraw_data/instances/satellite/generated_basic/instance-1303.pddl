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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	infrared2 - mode
	infrared0 - mode
	image1 - mode
	star5 - direction
	groundstation4 - direction
	star1 - direction
	groundstation3 - direction
	star0 - direction
	star2 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared2)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 star1)
	(supports instrument5 infrared2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument6 infrared0)
	(supports instrument6 infrared2)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared2)
	(supports instrument7 image1)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
)
(:goal (and
	(pointing satellite1 star7)
	(have_image star6 infrared0)
	(have_image star7 infrared0)
	(have_image planet8 infrared0)
	(have_image planet9 image1)
))

)
