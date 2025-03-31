(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	spectrograph0 - mode
	image1 - mode
	image2 - mode
	groundstation2 - direction
	star3 - direction
	groundstation5 - direction
	groundstation1 - direction
	star4 - direction
	groundstation0 - direction
	star6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star4)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(supports instrument1 image2)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(supports instrument4 image1)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
)
(:goal (and
	(have_image star6 image1)
	(have_image phenomenon7 image1)
	(have_image star8 spectrograph0)
))

)
