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
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	spectrograph2 - mode
	infrared0 - mode
	infrared1 - mode
	star1 - direction
	star2 - direction
	star3 - direction
	groundstation5 - direction
	groundstation4 - direction
	groundstation0 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star1)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 infrared0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star3)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star3)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation5)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
	(supports instrument8 spectrograph2)
	(supports instrument8 infrared0)
	(calibration_target instrument8 star2)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument10 spectrograph2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star2)
	(supports instrument11 spectrograph2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star2)
	(calibration_target instrument11 groundstation4)
	(supports instrument12 infrared1)
	(supports instrument12 infrared0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star3)
	(calibration_target instrument12 star2)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 groundstation4)
	(calibration_target instrument13 groundstation5)
	(supports instrument14 infrared0)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
)
(:goal (and
	(pointing satellite1 star2)
	(have_image planet6 spectrograph2)
	(have_image planet7 spectrograph2)
	(have_image star8 infrared1)
	(have_image star9 infrared1)
))

)
