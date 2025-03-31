(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot25 shot206 - shot
      ingredient39 ingredient488 ingredient165 ingredient36 - ingredient
      cocktail284 - cocktail
      dispenser488 dispenser68 dispenser440 dispenser86 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot25)
  (ontable shot206)
  (dispenses dispenser488 ingredient39)
  (dispenses dispenser68 ingredient488)
  (dispenses dispenser440 ingredient165)
  (dispenses dispenser86 ingredient36)
  (clean shaker86)
  (clean shot25)
  (clean shot206)
  (empty shaker86)
  (empty shot25)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail284 ingredient165)
  (cocktail-part2 cocktail284 ingredient36)
)
 (:goal
  (and
      (contains shot25 cocktail284)
)))
