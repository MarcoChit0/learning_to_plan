(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	infrared0 - mode
	image1 - mode
	infrared2 - mode
	groundstation0 - direction
	star5 - direction
	groundstation2 - direction
	groundstation1 - direction
	groundstation4 - direction
	star3 - direction
	star6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
	star10 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 star3)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite1 star7)
	(have_image star6 infrared2)
	(have_image star7 image1)
	(have_image star8 infrared2)
	(have_image planet9 infrared2)
	(have_image star10 infrared2)
))

)
