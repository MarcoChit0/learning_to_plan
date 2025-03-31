(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	spectrograph1 - mode
	image0 - mode
	thermograph2 - mode
	star0 - direction
	groundstation1 - direction
	star5 - direction
	star3 - direction
	star4 - direction
	groundstation2 - direction
	star6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star3)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation2)
	(calibration_target instrument3 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument4 spectrograph1)
	(supports instrument4 thermograph2)
	(supports instrument4 image0)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 star4)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
)
(:goal (and
	(have_image star6 image0)
	(have_image star7 thermograph2)
	(have_image star8 spectrograph1)
))

)
