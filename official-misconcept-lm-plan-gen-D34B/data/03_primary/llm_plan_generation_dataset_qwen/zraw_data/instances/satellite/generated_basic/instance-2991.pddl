(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	thermograph0 - mode
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	groundstation0 - direction
	groundstation5 - direction
	groundstation4 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 star2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 groundstation5)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 groundstation5)
	(have_image planet6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
