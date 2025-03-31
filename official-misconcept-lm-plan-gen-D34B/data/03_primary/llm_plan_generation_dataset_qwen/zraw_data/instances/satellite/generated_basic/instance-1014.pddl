(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	image0 - mode
	groundstation0 - direction
	groundstation1 - direction
	planet2 - direction
	planet3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation1)
	(on_board instrument4 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet2)
)
(:goal (and
	(pointing satellite0 planet2)
	(pointing satellite2 groundstation1)
	(have_image planet2 image0)
	(have_image planet3 image0)
	(have_image phenomenon4 image0)
))

)
