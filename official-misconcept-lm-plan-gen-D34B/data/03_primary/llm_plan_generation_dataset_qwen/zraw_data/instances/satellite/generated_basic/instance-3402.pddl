(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	satellite3 - satellite
	instrument3 - instrument
	infrared2 - mode
	thermograph1 - mode
	image0 - mode
	groundstation3 - direction
	star2 - direction
	star1 - direction
	groundstation0 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(pointing satellite2 star2)
	(pointing satellite3 groundstation0)
	(have_image phenomenon4 thermograph1)
))

)
