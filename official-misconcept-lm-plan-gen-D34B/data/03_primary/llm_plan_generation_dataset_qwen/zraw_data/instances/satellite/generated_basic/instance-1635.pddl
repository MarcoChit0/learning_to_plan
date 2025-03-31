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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	infrared2 - mode
	image0 - mode
	infrared1 - mode
	groundstation1 - direction
	star0 - direction
	planet2 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared2)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 infrared1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite1 planet2)
	(pointing satellite2 planet2)
	(have_image planet2 infrared2)
	(have_image planet3 infrared1)
	(have_image phenomenon4 infrared1)
))

)
