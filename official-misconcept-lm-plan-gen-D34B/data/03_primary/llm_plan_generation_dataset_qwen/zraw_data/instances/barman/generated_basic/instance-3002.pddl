(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot4 shot148 - shot
      ingredient309 ingredient99 ingredient465 - ingredient
      cocktail272 - cocktail
      dispenser249 dispenser186 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot4)
  (ontable shot148)
  (dispenses dispenser249 ingredient309)
  (dispenses dispenser186 ingredient99)
  (dispenses dispenser109 ingredient465)
  (clean shaker19)
  (clean shot4)
  (clean shot148)
  (empty shaker19)
  (empty shot4)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail272 ingredient309)
  (cocktail-part2 cocktail272 ingredient465)
)
 (:goal
  (and
      (contains shot4 cocktail272)
)))
