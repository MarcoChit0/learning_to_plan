(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	spectrograph2 - mode
	spectrograph3 - mode
	infrared0 - mode
	image1 - mode
	star2 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation1 - direction
	star3 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph3)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image planet5 spectrograph2)
	(have_image star6 spectrograph3)
	(have_image phenomenon7 infrared0)
	(have_image star8 spectrograph3)
))

)
