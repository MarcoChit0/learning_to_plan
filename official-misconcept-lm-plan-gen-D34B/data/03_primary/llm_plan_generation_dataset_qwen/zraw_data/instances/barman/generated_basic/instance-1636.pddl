(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot406 shot322 shot59 - shot
      ingredient455 ingredient283 - ingredient
      cocktail304 - cocktail
      dispenser28 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot406)
  (ontable shot322)
  (ontable shot59)
  (dispenses dispenser28 ingredient455)
  (dispenses dispenser217 ingredient283)
  (clean shaker305)
  (clean shot406)
  (clean shot322)
  (clean shot59)
  (empty shaker305)
  (empty shot406)
  (empty shot322)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail304 ingredient455)
  (cocktail-part2 cocktail304 ingredient283)
)
 (:goal
  (and
      (contains shot406 cocktail304)
      (contains shot322 cocktail304)
)))
