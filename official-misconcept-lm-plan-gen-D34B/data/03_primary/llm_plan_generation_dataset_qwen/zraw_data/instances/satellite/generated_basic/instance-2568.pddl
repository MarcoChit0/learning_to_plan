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
	image2 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
	groundstation4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 image2)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image phenomenon5 infrared0)
))

)
