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
	thermograph0 - mode
	groundstation4 - direction
	star5 - direction
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	star3 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star5)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image planet6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image star8 thermograph0)
))

)
