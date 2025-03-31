(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph0 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	groundstation4 - direction
	groundstation2 - direction
	star5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 groundstation2)
	(have_image star5 thermograph0)
	(have_image phenomenon6 thermograph0)
	(have_image phenomenon7 infrared1)
	(have_image phenomenon8 infrared1)
))

)
