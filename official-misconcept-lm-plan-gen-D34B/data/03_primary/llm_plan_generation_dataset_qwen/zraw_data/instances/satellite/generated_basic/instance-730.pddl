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
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	infrared1 - mode
	image2 - mode
	spectrograph0 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	groundstation5 - direction
	groundstation1 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image2)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star2)
	(supports instrument2 image2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument4 spectrograph0)
	(supports instrument4 image2)
	(supports instrument4 infrared1)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image2)
	(supports instrument5 infrared1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 infrared1)
	(supports instrument6 image2)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 image2)
	(calibration_target instrument8 groundstation4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument9 spectrograph0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 star2)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph0)
	(supports instrument10 infrared1)
	(supports instrument10 image2)
	(calibration_target instrument10 groundstation4)
	(supports instrument11 spectrograph0)
	(supports instrument11 image2)
	(calibration_target instrument11 groundstation4)
	(calibration_target instrument11 groundstation5)
	(supports instrument12 infrared1)
	(supports instrument12 image2)
	(calibration_target instrument12 star2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument13 spectrograph0)
	(supports instrument13 image2)
	(supports instrument13 infrared1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image2)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation5)
	(supports instrument15 infrared1)
	(supports instrument15 spectrograph0)
	(supports instrument15 image2)
	(calibration_target instrument15 groundstation1)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
)
(:goal (and
	(have_image star6 spectrograph0)
	(have_image star7 infrared1)
	(have_image planet8 image2)
	(have_image phenomenon9 spectrograph0)
))

)
