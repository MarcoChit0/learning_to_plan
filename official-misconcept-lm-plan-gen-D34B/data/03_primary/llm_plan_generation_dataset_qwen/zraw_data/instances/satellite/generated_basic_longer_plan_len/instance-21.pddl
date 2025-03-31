(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared2 - mode
	image0 - mode
	image1 - mode
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	star3 - direction
	phenomenon4 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image1)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 star3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
)
(:goal (and
	(pointing satellite0 planet5)
	(pointing satellite1 groundstation1)
	(pointing satellite2 planet5)
	(have_image phenomenon4 infrared2)
	(have_image planet5 image1)
	(have_image star6 infrared2)
	(have_image star7 infrared2)
	(have_image phenomenon8 infrared2)
	(have_image planet9 image0)
))

)
