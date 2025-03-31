(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared0 - mode
	image1 - mode
	infrared2 - mode
	infrared3 - mode
	star1 - direction
	groundstation0 - direction
	planet2 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 infrared3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument2 infrared3)
	(supports instrument2 infrared2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
)
(:goal (and
	(pointing satellite0 planet2)
	(have_image planet2 image1)
))

)
