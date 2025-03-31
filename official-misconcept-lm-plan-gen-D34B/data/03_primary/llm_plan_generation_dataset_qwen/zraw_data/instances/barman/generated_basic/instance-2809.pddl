(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot240 shot117 - shot
      ingredient342 ingredient291 - ingredient
      cocktail338 - cocktail
      dispenser498 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot240)
  (ontable shot117)
  (dispenses dispenser498 ingredient342)
  (dispenses dispenser14 ingredient291)
  (clean shaker454)
  (clean shot240)
  (clean shot117)
  (empty shaker454)
  (empty shot240)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail338 ingredient291)
  (cocktail-part2 cocktail338 ingredient342)
)
 (:goal
  (and
      (contains shot240 cocktail338)
)))
