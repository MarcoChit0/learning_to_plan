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
	spectrograph2 - mode
	thermograph1 - mode
	spectrograph0 - mode
	star1 - direction
	groundstation2 - direction
	star3 - direction
	groundstation5 - direction
	groundstation4 - direction
	star0 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	phenomenon9 - direction
	star10 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(calibration_target instrument1 groundstation4)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image planet6 thermograph1)
	(have_image phenomenon7 spectrograph0)
	(have_image star8 spectrograph0)
	(have_image phenomenon9 spectrograph2)
	(have_image star10 thermograph1)
))

)
