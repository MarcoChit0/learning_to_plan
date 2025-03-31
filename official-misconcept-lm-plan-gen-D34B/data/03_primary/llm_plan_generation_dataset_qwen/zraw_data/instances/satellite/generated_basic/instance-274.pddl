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
	thermograph1 - mode
	thermograph0 - mode
	star1 - direction
	star2 - direction
	groundstation0 - direction
	star3 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star2)
	(have_image planet4 thermograph0)
	(have_image star5 thermograph1)
	(have_image phenomenon6 thermograph0)
))

)
