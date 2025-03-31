(define (problem prob)
 (:domain barman)
 (:objects 
      shaker410 - shaker
      left right - hand
      shot64 shot83 - shot
      ingredient171 ingredient86 ingredient447 - ingredient
      cocktail1 - cocktail
      dispenser420 dispenser118 dispenser182 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker410)
  (ontable shot64)
  (ontable shot83)
  (dispenses dispenser420 ingredient171)
  (dispenses dispenser118 ingredient86)
  (dispenses dispenser182 ingredient447)
  (clean shaker410)
  (clean shot64)
  (clean shot83)
  (empty shaker410)
  (empty shot64)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker410 l0)
  (shaker-level shaker410 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient447)
  (cocktail-part2 cocktail1 ingredient86)
)
 (:goal
  (and
      (contains shot64 cocktail1)
)))
