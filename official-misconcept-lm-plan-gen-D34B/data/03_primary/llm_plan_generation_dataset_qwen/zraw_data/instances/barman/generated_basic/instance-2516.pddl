(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot350 - shot
      ingredient198 ingredient32 - ingredient
      cocktail293 - cocktail
      dispenser437 dispenser476 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot350)
  (dispenses dispenser437 ingredient198)
  (dispenses dispenser476 ingredient32)
  (clean shaker204)
  (clean shot350)
  (empty shaker204)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient32)
  (cocktail-part2 cocktail293 ingredient198)
)
 (:goal
  (and
      (contains shot350 cocktail293)
)))
