(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot220 shot278 - shot
      ingredient219 ingredient287 ingredient184 - ingredient
      cocktail364 - cocktail
      dispenser266 dispenser363 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot220)
  (ontable shot278)
  (dispenses dispenser266 ingredient219)
  (dispenses dispenser363 ingredient287)
  (dispenses dispenser231 ingredient184)
  (clean shaker485)
  (clean shot220)
  (clean shot278)
  (empty shaker485)
  (empty shot220)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail364 ingredient184)
  (cocktail-part2 cocktail364 ingredient287)
)
 (:goal
  (and
      (contains shot220 cocktail364)
)))
