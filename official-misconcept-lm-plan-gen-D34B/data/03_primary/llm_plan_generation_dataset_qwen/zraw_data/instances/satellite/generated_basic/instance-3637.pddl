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
	satellite3 - satellite
	instrument7 - instrument
	image1 - mode
	image0 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument7 image0)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite0 planet3)
	(pointing satellite1 planet3)
	(have_image phenomenon2 image0)
	(have_image planet3 image0)
))

)
