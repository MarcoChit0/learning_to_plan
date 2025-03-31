(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared2 - mode
	thermograph1 - mode
	spectrograph3 - mode
	spectrograph0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation4 - direction
	groundstation1 - direction
	groundstation3 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image star5 spectrograph3)
	(have_image star6 spectrograph3)
	(have_image phenomenon7 spectrograph3)
	(have_image star8 spectrograph3)
))

)
