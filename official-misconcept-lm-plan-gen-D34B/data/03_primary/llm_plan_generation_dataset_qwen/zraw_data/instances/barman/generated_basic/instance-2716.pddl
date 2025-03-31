(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot169 shot63 shot66 - shot
      ingredient284 ingredient155 ingredient363 - ingredient
      cocktail19 - cocktail
      dispenser174 dispenser463 dispenser495 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot169)
  (ontable shot63)
  (ontable shot66)
  (dispenses dispenser174 ingredient284)
  (dispenses dispenser463 ingredient155)
  (dispenses dispenser495 ingredient363)
  (clean shaker244)
  (clean shot169)
  (clean shot63)
  (clean shot66)
  (empty shaker244)
  (empty shot169)
  (empty shot63)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail19 ingredient284)
  (cocktail-part2 cocktail19 ingredient363)
)
 (:goal
  (and
      (contains shot169 cocktail19)
      (contains shot63 cocktail19)
)))
