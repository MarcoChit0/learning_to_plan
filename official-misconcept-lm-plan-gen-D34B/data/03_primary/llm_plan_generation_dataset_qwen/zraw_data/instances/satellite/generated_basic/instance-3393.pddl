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
	image1 - mode
	thermograph0 - mode
	image2 - mode
	groundstation0 - direction
	star3 - direction
	groundstation2 - direction
	groundstation1 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image2)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation1)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet4)
)
(:goal (and
	(pointing satellite0 planet4)
	(pointing satellite1 planet4)
	(pointing satellite3 groundstation0)
	(have_image planet4 image1)
))

)
