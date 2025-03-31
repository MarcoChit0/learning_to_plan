(define (problem prob)
 (:domain barman)
 (:objects 
      shaker265 - shaker
      left right - hand
      shot49 shot169 - shot
      ingredient226 ingredient349 - ingredient
      cocktail101 - cocktail
      dispenser323 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker265)
  (ontable shot49)
  (ontable shot169)
  (dispenses dispenser323 ingredient226)
  (dispenses dispenser133 ingredient349)
  (clean shaker265)
  (clean shot49)
  (clean shot169)
  (empty shaker265)
  (empty shot49)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker265 l0)
  (shaker-level shaker265 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient226)
  (cocktail-part2 cocktail101 ingredient349)
)
 (:goal
  (and
      (contains shot49 cocktail101)
)))
