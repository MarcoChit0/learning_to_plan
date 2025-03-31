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
	satellite2 - satellite
	instrument5 - instrument
	thermograph2 - mode
	thermograph0 - mode
	spectrograph1 - mode
	star1 - direction
	star0 - direction
	phenomenon2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 star3)
	(have_image phenomenon2 spectrograph1)
	(have_image star3 thermograph2)
	(have_image planet4 thermograph0)
))

)
