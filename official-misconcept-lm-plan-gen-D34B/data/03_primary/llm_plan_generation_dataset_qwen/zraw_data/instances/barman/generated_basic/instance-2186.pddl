(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot182 shot105 shot328 - shot
      ingredient234 ingredient247 - ingredient
      cocktail440 - cocktail
      dispenser174 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot182)
  (ontable shot105)
  (ontable shot328)
  (dispenses dispenser174 ingredient234)
  (dispenses dispenser229 ingredient247)
  (clean shaker185)
  (clean shot182)
  (clean shot105)
  (clean shot328)
  (empty shaker185)
  (empty shot182)
  (empty shot105)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail440 ingredient247)
  (cocktail-part2 cocktail440 ingredient234)
)
 (:goal
  (and
      (contains shot182 cocktail440)
      (contains shot105 ingredient247)
)))
