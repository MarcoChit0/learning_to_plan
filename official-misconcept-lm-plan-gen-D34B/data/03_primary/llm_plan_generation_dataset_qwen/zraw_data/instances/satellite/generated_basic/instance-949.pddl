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
	satellite2 - satellite
	instrument5 - instrument
	spectrograph3 - mode
	image1 - mode
	infrared2 - mode
	image0 - mode
	star3 - direction
	star1 - direction
	groundstation2 - direction
	groundstation0 - direction
	planet4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph3)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 spectrograph3)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph3)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(supports instrument4 image0)
	(supports instrument4 spectrograph3)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 spectrograph3)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite1 star3)
	(pointing satellite2 star6)
	(have_image planet4 image1)
	(have_image planet5 image0)
	(have_image star6 image0)
))

)
