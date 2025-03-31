(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot92 shot38 - shot
      ingredient328 ingredient477 ingredient386 ingredient169 - ingredient
      cocktail401 - cocktail
      dispenser398 dispenser121 dispenser193 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot92)
  (ontable shot38)
  (dispenses dispenser398 ingredient328)
  (dispenses dispenser121 ingredient477)
  (dispenses dispenser193 ingredient386)
  (dispenses dispenser44 ingredient169)
  (clean shaker63)
  (clean shot92)
  (clean shot38)
  (empty shaker63)
  (empty shot92)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail401 ingredient328)
  (cocktail-part2 cocktail401 ingredient386)
)
 (:goal
  (and
      (contains shot92 cocktail401)
)))
