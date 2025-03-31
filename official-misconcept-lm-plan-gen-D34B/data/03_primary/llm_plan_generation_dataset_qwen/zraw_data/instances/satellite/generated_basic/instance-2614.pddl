(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	spectrograph0 - mode
	infrared2 - mode
	thermograph3 - mode
	thermograph1 - mode
	star0 - direction
	groundstation1 - direction
	planet2 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph0)
	(supports instrument2 thermograph3)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet2)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet2 spectrograph0)
))

)
