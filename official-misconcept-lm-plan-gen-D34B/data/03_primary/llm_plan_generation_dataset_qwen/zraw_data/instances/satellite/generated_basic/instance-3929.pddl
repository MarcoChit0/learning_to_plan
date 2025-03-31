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
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	infrared1 - mode
	spectrograph0 - mode
	image2 - mode
	groundstation0 - direction
	star5 - direction
	star4 - direction
	star1 - direction
	star3 - direction
	groundstation2 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star4)
	(supports instrument1 image2)
	(calibration_target instrument1 star1)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(calibration_target instrument2 star5)
	(supports instrument3 image2)
	(calibration_target instrument3 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument4 image2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph0)
	(supports instrument5 image2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph0)
	(supports instrument6 image2)
	(calibration_target instrument6 star1)
	(calibration_target instrument6 star4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation2)
	(calibration_target instrument7 star3)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite2 planet6)
	(have_image planet6 infrared1)
))

)
