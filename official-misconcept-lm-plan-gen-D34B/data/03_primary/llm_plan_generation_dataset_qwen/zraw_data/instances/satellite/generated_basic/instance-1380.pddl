(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
)
(:goal (and
	(pointing satellite1 groundstation1)
	(have_image phenomenon2 thermograph1)
	(have_image planet3 infrared0)
))

)
