(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot498 shot45 - shot
      ingredient427 ingredient407 ingredient290 - ingredient
      cocktail284 - cocktail
      dispenser46 dispenser217 dispenser347 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot498)
  (ontable shot45)
  (dispenses dispenser46 ingredient427)
  (dispenses dispenser217 ingredient407)
  (dispenses dispenser347 ingredient290)
  (clean shaker45)
  (clean shot498)
  (clean shot45)
  (empty shaker45)
  (empty shot498)
  (empty shot45)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail284 ingredient407)
  (cocktail-part2 cocktail284 ingredient427)
)
 (:goal
  (and
      (contains shot498 cocktail284)
)))
