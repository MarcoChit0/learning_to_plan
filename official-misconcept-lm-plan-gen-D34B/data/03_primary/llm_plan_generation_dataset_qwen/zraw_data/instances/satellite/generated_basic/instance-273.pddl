(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	infrared1 - mode
	thermograph0 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(have_image star4 infrared1)
	(have_image phenomenon5 thermograph0)
	(have_image planet6 thermograph0)
))

)
