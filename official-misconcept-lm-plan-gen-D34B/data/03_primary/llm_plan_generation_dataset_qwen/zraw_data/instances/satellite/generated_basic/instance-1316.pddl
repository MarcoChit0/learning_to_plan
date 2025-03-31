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
	infrared2 - mode
	thermograph1 - mode
	thermograph0 - mode
	star0 - direction
	groundstation5 - direction
	star4 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 infrared2)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star4)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 star4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument5 thermograph1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite1 groundstation5)
	(pointing satellite2 planet6)
	(have_image planet6 thermograph0)
	(have_image planet7 infrared2)
	(have_image phenomenon8 infrared2)
	(have_image phenomenon9 infrared2)
))

)
