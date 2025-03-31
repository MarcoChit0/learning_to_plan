(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	image1 - mode
	image0 - mode
	groundstation2 - direction
	star3 - direction
	groundstation4 - direction
	star0 - direction
	star5 - direction
	star1 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image0)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
)
(:goal (and
	(have_image planet6 image1)
	(have_image planet7 image1)
	(have_image planet8 image1)
))

)
