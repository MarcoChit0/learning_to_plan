(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot433 shot402 - shot
      ingredient147 ingredient419 ingredient133 - ingredient
      cocktail155 - cocktail
      dispenser11 dispenser448 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot433)
  (ontable shot402)
  (dispenses dispenser11 ingredient147)
  (dispenses dispenser448 ingredient419)
  (dispenses dispenser102 ingredient133)
  (clean shaker233)
  (clean shot433)
  (clean shot402)
  (empty shaker233)
  (empty shot433)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient133)
  (cocktail-part2 cocktail155 ingredient147)
)
 (:goal
  (and
      (contains shot433 cocktail155)
)))
