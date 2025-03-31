(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot381 shot143 - shot
      ingredient404 ingredient155 ingredient337 ingredient69 - ingredient
      cocktail7 - cocktail
      dispenser341 dispenser437 dispenser312 dispenser99 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot381)
  (ontable shot143)
  (dispenses dispenser341 ingredient404)
  (dispenses dispenser437 ingredient155)
  (dispenses dispenser312 ingredient337)
  (dispenses dispenser99 ingredient69)
  (clean shaker146)
  (clean shot381)
  (clean shot143)
  (empty shaker146)
  (empty shot381)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail7 ingredient337)
  (cocktail-part2 cocktail7 ingredient404)
)
 (:goal
  (and
      (contains shot381 cocktail7)
)))
