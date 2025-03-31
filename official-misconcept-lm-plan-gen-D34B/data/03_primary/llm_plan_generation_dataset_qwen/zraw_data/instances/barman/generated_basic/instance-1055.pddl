(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot224 shot474 shot36 - shot
      ingredient251 ingredient121 ingredient233 ingredient203 - ingredient
      cocktail1 - cocktail
      dispenser80 dispenser196 dispenser62 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot224)
  (ontable shot474)
  (ontable shot36)
  (dispenses dispenser80 ingredient251)
  (dispenses dispenser196 ingredient121)
  (dispenses dispenser62 ingredient233)
  (dispenses dispenser204 ingredient203)
  (clean shaker497)
  (clean shot224)
  (clean shot474)
  (clean shot36)
  (empty shaker497)
  (empty shot224)
  (empty shot474)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient121)
  (cocktail-part2 cocktail1 ingredient233)
)
 (:goal
  (and
      (contains shot224 cocktail1)
      (contains shot474 cocktail1)
)))
