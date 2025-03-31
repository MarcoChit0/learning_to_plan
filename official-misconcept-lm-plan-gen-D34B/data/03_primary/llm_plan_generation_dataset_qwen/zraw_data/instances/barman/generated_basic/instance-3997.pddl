(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot114 shot67 shot159 - shot
      ingredient134 ingredient419 - ingredient
      cocktail308 - cocktail
      dispenser282 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot114)
  (ontable shot67)
  (ontable shot159)
  (dispenses dispenser282 ingredient134)
  (dispenses dispenser249 ingredient419)
  (clean shaker24)
  (clean shot114)
  (clean shot67)
  (clean shot159)
  (empty shaker24)
  (empty shot114)
  (empty shot67)
  (empty shot159)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient134)
  (cocktail-part2 cocktail308 ingredient419)
)
 (:goal
  (and
      (contains shot114 cocktail308)
      (contains shot67 cocktail308)
)))
