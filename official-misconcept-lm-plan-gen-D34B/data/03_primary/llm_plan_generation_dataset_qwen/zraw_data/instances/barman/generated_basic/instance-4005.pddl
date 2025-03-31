(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot282 shot333 shot494 - shot
      ingredient477 ingredient127 - ingredient
      cocktail235 - cocktail
      dispenser182 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot282)
  (ontable shot333)
  (ontable shot494)
  (dispenses dispenser182 ingredient477)
  (dispenses dispenser364 ingredient127)
  (clean shaker108)
  (clean shot282)
  (clean shot333)
  (clean shot494)
  (empty shaker108)
  (empty shot282)
  (empty shot333)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail235 ingredient127)
  (cocktail-part2 cocktail235 ingredient477)
)
 (:goal
  (and
      (contains shot282 cocktail235)
      (contains shot333 cocktail235)
)))
