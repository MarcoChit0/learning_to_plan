(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	thermograph0 - mode
	star1 - direction
	star0 - direction
	star4 - direction
	groundstation3 - direction
	groundstation2 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star4)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star4)
	(calibration_target instrument2 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star4)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image star6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
