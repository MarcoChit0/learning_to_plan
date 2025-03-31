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
	satellite2 - satellite
	instrument5 - instrument
	spectrograph1 - mode
	thermograph2 - mode
	spectrograph0 - mode
	groundstation0 - direction
	star1 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation2 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 star5)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 star5)
	(supports instrument2 spectrograph1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 groundstation3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument5 thermograph2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 groundstation2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(have_image star6 thermograph2)
	(have_image phenomenon7 spectrograph1)
	(have_image phenomenon8 thermograph2)
	(have_image star9 spectrograph1)
))

)
