(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	image1 - mode
	infrared0 - mode
	groundstation1 - direction
	groundstation0 - direction
	planet2 - direction
	planet3 - direction
	planet4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(pointing satellite1 planet3)
	(have_image planet2 image1)
	(have_image planet3 infrared0)
	(have_image planet4 infrared0)
	(have_image star5 image1)
))

)
