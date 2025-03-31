(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot48 shot281 - shot
      ingredient196 ingredient195 ingredient140 ingredient170 - ingredient
      cocktail426 - cocktail
      dispenser205 dispenser376 dispenser69 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot48)
  (ontable shot281)
  (dispenses dispenser205 ingredient196)
  (dispenses dispenser376 ingredient195)
  (dispenses dispenser69 ingredient140)
  (dispenses dispenser143 ingredient170)
  (clean shaker63)
  (clean shot48)
  (clean shot281)
  (empty shaker63)
  (empty shot48)
  (empty shot281)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail426 ingredient140)
  (cocktail-part2 cocktail426 ingredient170)
)
 (:goal
  (and
      (contains shot48 cocktail426)
)))
