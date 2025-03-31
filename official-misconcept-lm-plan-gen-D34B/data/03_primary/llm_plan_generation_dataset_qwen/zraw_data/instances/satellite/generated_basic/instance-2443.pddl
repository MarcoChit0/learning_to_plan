(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	infrared1 - mode
	spectrograph0 - mode
	spectrograph2 - mode
	star0 - direction
	star1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
	star4 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
)
(:goal (and
	(have_image phenomenon2 infrared1)
	(have_image phenomenon3 infrared1)
	(have_image star4 infrared1)
	(have_image star5 spectrograph2)
))

)
