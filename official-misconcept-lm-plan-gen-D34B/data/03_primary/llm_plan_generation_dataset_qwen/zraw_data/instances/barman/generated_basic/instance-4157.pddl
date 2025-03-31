(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot82 shot337 - shot
      ingredient8 ingredient492 - ingredient
      cocktail480 - cocktail
      dispenser133 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot82)
  (ontable shot337)
  (dispenses dispenser133 ingredient8)
  (dispenses dispenser384 ingredient492)
  (clean shaker464)
  (clean shot82)
  (clean shot337)
  (empty shaker464)
  (empty shot82)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient492)
  (cocktail-part2 cocktail480 ingredient8)
)
 (:goal
  (and
      (contains shot82 cocktail480)
)))
