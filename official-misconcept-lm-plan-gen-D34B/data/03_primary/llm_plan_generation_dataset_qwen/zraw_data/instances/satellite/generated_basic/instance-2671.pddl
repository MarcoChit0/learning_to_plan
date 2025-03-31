(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	thermograph3 - mode
	thermograph1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	star1 - direction
	groundstation2 - direction
	star0 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph3)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 star3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star0)
	(have_image planet4 thermograph0)
))

)
