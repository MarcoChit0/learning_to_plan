(define (problem prob)
 (:domain barman)
 (:objects 
      shaker175 - shaker
      left right - hand
      shot61 - shot
      ingredient93 ingredient436 ingredient196 ingredient353 - ingredient
      cocktail82 - cocktail
      dispenser414 dispenser31 dispenser344 dispenser42 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker175)
  (ontable shot61)
  (dispenses dispenser414 ingredient93)
  (dispenses dispenser31 ingredient436)
  (dispenses dispenser344 ingredient196)
  (dispenses dispenser42 ingredient353)
  (clean shaker175)
  (clean shot61)
  (empty shaker175)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker175 l0)
  (shaker-level shaker175 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient353)
  (cocktail-part2 cocktail82 ingredient196)
)
 (:goal
  (and
      (contains shot61 cocktail82)
)))
