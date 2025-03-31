(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image0 - mode
	image2 - mode
	infrared1 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	groundstation4 - direction
	star3 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared1)
	(supports instrument1 image0)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 image2)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet5 infrared1)
	(have_image star6 infrared1)
	(have_image star7 image2)
	(have_image planet8 infrared1)
	(have_image planet9 infrared1)
))

)
