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
	thermograph2 - mode
	thermograph3 - mode
	thermograph1 - mode
	spectrograph0 - mode
	star3 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph3)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph2)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
)
(:goal (and
	(have_image planet5 thermograph2)
	(have_image phenomenon6 spectrograph0)
	(have_image phenomenon7 thermograph3)
	(have_image star8 thermograph1)
))

)
