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
	infrared0 - mode
	image1 - mode
	image2 - mode
	groundstation1 - direction
	groundstation0 - direction
	phenomenon2 - direction
	planet3 - direction
	phenomenon4 - direction
	star5 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
)
(:goal (and
	(pointing satellite1 phenomenon4)
	(have_image phenomenon2 infrared0)
	(have_image planet3 image2)
	(have_image phenomenon4 image1)
	(have_image star5 image1)
))

)
