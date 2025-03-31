(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot326 shot233 shot139 - shot
      ingredient155 ingredient261 - ingredient
      cocktail209 - cocktail
      dispenser465 dispenser499 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot326)
  (ontable shot233)
  (ontable shot139)
  (dispenses dispenser465 ingredient155)
  (dispenses dispenser499 ingredient261)
  (clean shaker143)
  (clean shot326)
  (clean shot233)
  (clean shot139)
  (empty shaker143)
  (empty shot326)
  (empty shot233)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail209 ingredient261)
  (cocktail-part2 cocktail209 ingredient155)
)
 (:goal
  (and
      (contains shot326 cocktail209)
      (contains shot233 ingredient261)
)))
