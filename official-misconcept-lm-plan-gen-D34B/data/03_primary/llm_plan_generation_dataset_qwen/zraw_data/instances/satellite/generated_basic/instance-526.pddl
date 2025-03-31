(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	infrared1 - mode
	image2 - mode
	image0 - mode
	infrared3 - mode
	groundstation2 - direction
	star3 - direction
	star0 - direction
	star1 - direction
	groundstation4 - direction
	phenomenon5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(supports instrument2 image2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image phenomenon5 infrared1)
	(have_image star6 image2)
	(have_image planet7 infrared1)
	(have_image phenomenon8 image2)
))

)
