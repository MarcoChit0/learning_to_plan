(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot85 shot94 shot445 - shot
      ingredient495 ingredient303 ingredient444 ingredient16 - ingredient
      cocktail1 - cocktail
      dispenser196 dispenser488 dispenser60 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot85)
  (ontable shot94)
  (ontable shot445)
  (dispenses dispenser196 ingredient495)
  (dispenses dispenser488 ingredient303)
  (dispenses dispenser60 ingredient444)
  (dispenses dispenser102 ingredient16)
  (clean shaker110)
  (clean shot85)
  (clean shot94)
  (clean shot445)
  (empty shaker110)
  (empty shot85)
  (empty shot94)
  (empty shot445)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient16)
  (cocktail-part2 cocktail1 ingredient303)
)
 (:goal
  (and
      (contains shot85 cocktail1)
      (contains shot94 cocktail1)
)))
