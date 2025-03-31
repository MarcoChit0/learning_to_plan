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
	instrument5 - instrument
	thermograph1 - mode
	thermograph2 - mode
	image0 - mode
	thermograph3 - mode
	groundstation0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument3 thermograph3)
	(supports instrument3 thermograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph2)
	(supports instrument4 thermograph3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet3)
)
(:goal (and
	(pointing satellite1 phenomenon1)
	(have_image phenomenon1 thermograph3)
	(have_image phenomenon2 thermograph2)
	(have_image planet3 image0)
))

)
