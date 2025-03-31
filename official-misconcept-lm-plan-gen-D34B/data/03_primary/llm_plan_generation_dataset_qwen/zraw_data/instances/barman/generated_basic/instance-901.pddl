(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot311 shot207 - shot
      ingredient443 ingredient249 - ingredient
      cocktail1 - cocktail
      dispenser254 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot311)
  (ontable shot207)
  (dispenses dispenser254 ingredient443)
  (dispenses dispenser146 ingredient249)
  (clean shaker443)
  (clean shot311)
  (clean shot207)
  (empty shaker443)
  (empty shot311)
  (empty shot207)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient249)
  (cocktail-part2 cocktail1 ingredient443)
)
 (:goal
  (and
      (contains shot311 cocktail1)
)))
