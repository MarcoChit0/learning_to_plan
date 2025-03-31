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
	spectrograph0 - mode
	spectrograph1 - mode
	thermograph3 - mode
	thermograph2 - mode
	groundstation1 - direction
	star2 - direction
	star4 - direction
	star5 - direction
	groundstation0 - direction
	star3 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star5)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star3)
	(supports instrument2 thermograph3)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 star3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 star5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument5 spectrograph1)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image star6 thermograph3)
))

)
