(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot119 - shot
      ingredient475 ingredient327 ingredient452 - ingredient
      cocktail246 - cocktail
      dispenser75 dispenser499 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot119)
  (dispenses dispenser75 ingredient475)
  (dispenses dispenser499 ingredient327)
  (dispenses dispenser157 ingredient452)
  (clean shaker183)
  (clean shot119)
  (empty shaker183)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail246 ingredient475)
  (cocktail-part2 cocktail246 ingredient452)
)
 (:goal
  (and
      (contains shot119 cocktail246)
)))
