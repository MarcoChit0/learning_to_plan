(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	spectrograph0 - mode
	image2 - mode
	image3 - mode
	image1 - mode
	star0 - direction
	star3 - direction
	groundstation1 - direction
	groundstation2 - direction
	groundstation4 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument1 image1)
	(supports instrument1 image2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image3)
	(supports instrument2 image1)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
)
(:goal (and
	(have_image planet5 image3)
	(have_image star6 image1)
	(have_image planet7 image2)
	(have_image phenomenon8 image1)
))

)
