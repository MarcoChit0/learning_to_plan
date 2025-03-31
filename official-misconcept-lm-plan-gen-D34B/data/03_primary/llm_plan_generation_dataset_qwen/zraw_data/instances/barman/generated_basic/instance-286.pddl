(define (problem prob)
 (:domain barman)
 (:objects 
      shaker335 - shaker
      left right - hand
      shot425 shot434 - shot
      ingredient455 ingredient360 ingredient476 - ingredient
      cocktail1 - cocktail
      dispenser107 dispenser378 dispenser439 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker335)
  (ontable shot425)
  (ontable shot434)
  (dispenses dispenser107 ingredient455)
  (dispenses dispenser378 ingredient360)
  (dispenses dispenser439 ingredient476)
  (clean shaker335)
  (clean shot425)
  (clean shot434)
  (empty shaker335)
  (empty shot425)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker335 l0)
  (shaker-level shaker335 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient476)
  (cocktail-part2 cocktail1 ingredient455)
)
 (:goal
  (and
      (contains shot425 cocktail1)
)))
