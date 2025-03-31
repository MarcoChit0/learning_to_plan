(define (problem prob)
 (:domain barman)
 (:objects 
      shaker226 - shaker
      left right - hand
      shot394 - shot
      ingredient494 ingredient139 - ingredient
      cocktail224 - cocktail
      dispenser338 dispenser396 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker226)
  (ontable shot394)
  (dispenses dispenser338 ingredient494)
  (dispenses dispenser396 ingredient139)
  (clean shaker226)
  (clean shot394)
  (empty shaker226)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker226 l0)
  (shaker-level shaker226 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail224 ingredient494)
  (cocktail-part2 cocktail224 ingredient139)
)
 (:goal
  (and
      (contains shot394 cocktail224)
)))
