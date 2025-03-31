(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot94 shot229 - shot
      ingredient104 ingredient180 - ingredient
      cocktail1 - cocktail
      dispenser392 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot94)
  (ontable shot229)
  (dispenses dispenser392 ingredient104)
  (dispenses dispenser368 ingredient180)
  (clean shaker95)
  (clean shot94)
  (clean shot229)
  (empty shaker95)
  (empty shot94)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient104)
  (cocktail-part2 cocktail1 ingredient180)
)
 (:goal
  (and
      (contains shot94 cocktail1)
)))
