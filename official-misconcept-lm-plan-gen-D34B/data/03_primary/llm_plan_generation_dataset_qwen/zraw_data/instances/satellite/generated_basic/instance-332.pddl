(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	spectrograph3 - mode
	spectrograph0 - mode
	image1 - mode
	spectrograph2 - mode
	groundstation5 - direction
	groundstation0 - direction
	groundstation3 - direction
	star1 - direction
	star4 - direction
	groundstation2 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 image1)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument2 spectrograph3)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph3)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation2)
	(calibration_target instrument4 star4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(have_image star6 spectrograph2)
))

)
