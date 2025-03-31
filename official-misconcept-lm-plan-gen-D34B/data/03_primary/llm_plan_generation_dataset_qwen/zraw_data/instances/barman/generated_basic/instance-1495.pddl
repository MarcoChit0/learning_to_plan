(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot160 shot238 - shot
      ingredient179 ingredient24 ingredient473 ingredient265 - ingredient
      cocktail73 - cocktail
      dispenser13 dispenser260 dispenser195 dispenser373 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot160)
  (ontable shot238)
  (dispenses dispenser13 ingredient179)
  (dispenses dispenser260 ingredient24)
  (dispenses dispenser195 ingredient473)
  (dispenses dispenser373 ingredient265)
  (clean shaker432)
  (clean shot160)
  (clean shot238)
  (empty shaker432)
  (empty shot160)
  (empty shot238)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail73 ingredient24)
  (cocktail-part2 cocktail73 ingredient265)
)
 (:goal
  (and
      (contains shot160 cocktail73)
)))
