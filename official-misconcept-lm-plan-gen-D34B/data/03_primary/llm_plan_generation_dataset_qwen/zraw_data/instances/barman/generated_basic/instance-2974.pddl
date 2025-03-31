(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot46 - shot
      ingredient302 ingredient315 - ingredient
      cocktail209 - cocktail
      dispenser276 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot46)
  (dispenses dispenser276 ingredient302)
  (dispenses dispenser106 ingredient315)
  (clean shaker22)
  (clean shot46)
  (empty shaker22)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail209 ingredient302)
  (cocktail-part2 cocktail209 ingredient315)
)
 (:goal
  (and
      (contains shot46 cocktail209)
)))
