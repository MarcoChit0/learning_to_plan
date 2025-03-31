(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot426 shot336 shot182 - shot
      ingredient53 ingredient236 - ingredient
      cocktail302 - cocktail
      dispenser89 dispenser187 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot426)
  (ontable shot336)
  (ontable shot182)
  (dispenses dispenser89 ingredient53)
  (dispenses dispenser187 ingredient236)
  (clean shaker178)
  (clean shot426)
  (clean shot336)
  (clean shot182)
  (empty shaker178)
  (empty shot426)
  (empty shot336)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient53)
  (cocktail-part2 cocktail302 ingredient236)
)
 (:goal
  (and
      (contains shot426 cocktail302)
      (contains shot336 cocktail302)
)))
