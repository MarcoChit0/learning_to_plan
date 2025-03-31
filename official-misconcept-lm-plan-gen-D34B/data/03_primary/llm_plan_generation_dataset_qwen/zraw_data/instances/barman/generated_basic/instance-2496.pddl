(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot435 shot205 shot381 - shot
      ingredient186 ingredient357 ingredient25 - ingredient
      cocktail165 - cocktail
      dispenser209 dispenser494 dispenser160 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot435)
  (ontable shot205)
  (ontable shot381)
  (dispenses dispenser209 ingredient186)
  (dispenses dispenser494 ingredient357)
  (dispenses dispenser160 ingredient25)
  (clean shaker463)
  (clean shot435)
  (clean shot205)
  (clean shot381)
  (empty shaker463)
  (empty shot435)
  (empty shot205)
  (empty shot381)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient25)
  (cocktail-part2 cocktail165 ingredient186)
)
 (:goal
  (and
      (contains shot435 cocktail165)
      (contains shot205 cocktail165)
)))
