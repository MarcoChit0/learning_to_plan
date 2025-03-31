(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image3 - mode
	infrared0 - mode
	infrared2 - mode
	infrared1 - mode
	groundstation3 - direction
	groundstation4 - direction
	star1 - direction
	star2 - direction
	star0 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
	phenomenon10 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(supports instrument2 image3)
	(calibration_target instrument2 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet9)
)
(:goal (and
	(have_image star5 image3)
	(have_image phenomenon6 infrared2)
	(have_image planet7 image3)
	(have_image star8 infrared2)
	(have_image planet9 image3)
	(have_image phenomenon10 infrared2)
))

)
