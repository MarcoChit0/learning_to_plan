(define (problem prob)
 (:domain barman)
 (:objects 
      shaker407 - shaker
      left right - hand
      shot248 shot224 shot62 - shot
      ingredient163 ingredient252 - ingredient
      cocktail419 - cocktail
      dispenser81 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker407)
  (ontable shot248)
  (ontable shot224)
  (ontable shot62)
  (dispenses dispenser81 ingredient163)
  (dispenses dispenser471 ingredient252)
  (clean shaker407)
  (clean shot248)
  (clean shot224)
  (clean shot62)
  (empty shaker407)
  (empty shot248)
  (empty shot224)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker407 l0)
  (shaker-level shaker407 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient252)
  (cocktail-part2 cocktail419 ingredient163)
)
 (:goal
  (and
      (contains shot248 cocktail419)
      (contains shot224 cocktail419)
)))
