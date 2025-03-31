(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	thermograph2 - mode
	thermograph1 - mode
	thermograph3 - mode
	spectrograph0 - mode
	star3 - direction
	groundstation2 - direction
	groundstation0 - direction
	groundstation4 - direction
	groundstation1 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(have_image phenomenon5 thermograph2)
	(have_image planet6 thermograph2)
	(have_image planet7 thermograph3)
	(have_image phenomenon8 thermograph2)
))

)
