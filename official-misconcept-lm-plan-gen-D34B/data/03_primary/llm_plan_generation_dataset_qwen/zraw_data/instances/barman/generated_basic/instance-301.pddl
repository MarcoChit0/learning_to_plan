(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot331 shot85 - shot
      ingredient357 ingredient224 - ingredient
      cocktail1 - cocktail
      dispenser73 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot331)
  (ontable shot85)
  (dispenses dispenser73 ingredient357)
  (dispenses dispenser207 ingredient224)
  (clean shaker216)
  (clean shot331)
  (clean shot85)
  (empty shaker216)
  (empty shot331)
  (empty shot85)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient357)
  (cocktail-part2 cocktail1 ingredient224)
)
 (:goal
  (and
      (contains shot331 cocktail1)
)))
