(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot223 shot322 shot210 - shot
      ingredient309 ingredient351 ingredient406 - ingredient
      cocktail399 - cocktail
      dispenser51 dispenser292 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot223)
  (ontable shot322)
  (ontable shot210)
  (dispenses dispenser51 ingredient309)
  (dispenses dispenser292 ingredient351)
  (dispenses dispenser465 ingredient406)
  (clean shaker499)
  (clean shot223)
  (clean shot322)
  (clean shot210)
  (empty shaker499)
  (empty shot223)
  (empty shot322)
  (empty shot210)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient309)
  (cocktail-part2 cocktail399 ingredient406)
)
 (:goal
  (and
      (contains shot223 cocktail399)
      (contains shot322 cocktail399)
)))
