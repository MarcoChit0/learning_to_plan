(define (problem prob)
 (:domain barman)
 (:objects 
      shaker35 - shaker
      left right - hand
      shot481 shot119 - shot
      ingredient24 ingredient356 - ingredient
      cocktail1 - cocktail
      dispenser211 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker35)
  (ontable shot481)
  (ontable shot119)
  (dispenses dispenser211 ingredient24)
  (dispenses dispenser35 ingredient356)
  (clean shaker35)
  (clean shot481)
  (clean shot119)
  (empty shaker35)
  (empty shot481)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker35 l0)
  (shaker-level shaker35 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient356)
  (cocktail-part2 cocktail1 ingredient24)
)
 (:goal
  (and
      (contains shot481 cocktail1)
)))
