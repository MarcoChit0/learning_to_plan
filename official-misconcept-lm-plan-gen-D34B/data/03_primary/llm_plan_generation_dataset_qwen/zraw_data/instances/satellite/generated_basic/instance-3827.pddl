(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	image0 - mode
	star4 - direction
	groundstation2 - direction
	star0 - direction
	star3 - direction
	star5 - direction
	star1 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image0)
	(calibration_target instrument1 star5)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument3 image0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star5)
	(supports instrument4 image0)
	(calibration_target instrument4 star3)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(calibration_target instrument5 star5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite0 star5)
	(pointing satellite1 planet8)
	(have_image planet6 image0)
	(have_image star7 image0)
	(have_image planet8 image0)
))

)
