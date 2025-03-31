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
	satellite2 - satellite
	instrument5 - instrument
	infrared0 - mode
	thermograph1 - mode
	groundstation1 - direction
	groundstation5 - direction
	star6 - direction
	star0 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation2 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
	planet10 - direction
	planet11 - direction
	phenomenon12 - direction
	phenomenon13 - direction
	phenomenon14 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon12)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation2)
	(calibration_target instrument2 star6)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation3)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(have_image planet7 thermograph1)
	(have_image planet8 infrared0)
	(have_image planet9 infrared0)
	(have_image planet10 infrared0)
	(have_image planet11 thermograph1)
	(have_image phenomenon12 infrared0)
	(have_image phenomenon13 thermograph1)
	(have_image phenomenon14 infrared0)
))

)
