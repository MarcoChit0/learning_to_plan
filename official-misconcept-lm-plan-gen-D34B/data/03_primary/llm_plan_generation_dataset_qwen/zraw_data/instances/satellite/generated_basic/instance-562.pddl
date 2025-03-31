(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	image0 - mode
	infrared2 - mode
	image1 - mode
	groundstation1 - direction
	star0 - direction
	phenomenon2 - direction
	planet3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 image0)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon5)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
)
(:goal (and
	(pointing satellite0 planet6)
	(pointing satellite2 phenomenon5)
	(have_image phenomenon2 image1)
	(have_image planet3 infrared2)
	(have_image phenomenon4 image0)
	(have_image phenomenon5 infrared2)
	(have_image planet6 image1)
))

)
