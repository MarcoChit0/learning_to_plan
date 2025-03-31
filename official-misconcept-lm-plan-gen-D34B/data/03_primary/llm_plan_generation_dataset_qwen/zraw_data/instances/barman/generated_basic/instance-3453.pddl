(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot173 shot357 - shot
      ingredient355 ingredient198 ingredient233 ingredient229 - ingredient
      cocktail413 - cocktail
      dispenser81 dispenser343 dispenser397 dispenser339 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot173)
  (ontable shot357)
  (dispenses dispenser81 ingredient355)
  (dispenses dispenser343 ingredient198)
  (dispenses dispenser397 ingredient233)
  (dispenses dispenser339 ingredient229)
  (clean shaker352)
  (clean shot173)
  (clean shot357)
  (empty shaker352)
  (empty shot173)
  (empty shot357)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail413 ingredient198)
  (cocktail-part2 cocktail413 ingredient229)
)
 (:goal
  (and
      (contains shot173 cocktail413)
)))
