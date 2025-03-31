(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot403 shot224 - shot
      ingredient131 ingredient428 - ingredient
      cocktail13 - cocktail
      dispenser409 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot403)
  (ontable shot224)
  (dispenses dispenser409 ingredient131)
  (dispenses dispenser4 ingredient428)
  (clean shaker185)
  (clean shot403)
  (clean shot224)
  (empty shaker185)
  (empty shot403)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient131)
  (cocktail-part2 cocktail13 ingredient428)
)
 (:goal
  (and
      (contains shot403 cocktail13)
)))
