(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot187 shot57 - shot
      ingredient442 ingredient166 - ingredient
      cocktail1 - cocktail
      dispenser123 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot187)
  (ontable shot57)
  (dispenses dispenser123 ingredient442)
  (dispenses dispenser422 ingredient166)
  (clean shaker400)
  (clean shot187)
  (clean shot57)
  (empty shaker400)
  (empty shot187)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient442)
  (cocktail-part2 cocktail1 ingredient166)
)
 (:goal
  (and
      (contains shot187 cocktail1)
)))
