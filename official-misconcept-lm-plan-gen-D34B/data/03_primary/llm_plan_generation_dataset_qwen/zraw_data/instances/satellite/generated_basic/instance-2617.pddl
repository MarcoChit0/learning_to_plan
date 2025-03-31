(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	spectrograph2 - mode
	infrared1 - mode
	spectrograph0 - mode
	thermograph3 - mode
	groundstation0 - direction
	star1 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph0)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image phenomenon2 spectrograph0)
))

)
