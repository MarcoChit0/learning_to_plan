(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot306 shot425 shot488 - shot
      ingredient116 ingredient406 ingredient122 ingredient129 - ingredient
      cocktail156 - cocktail
      dispenser492 dispenser348 dispenser82 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot306)
  (ontable shot425)
  (ontable shot488)
  (dispenses dispenser492 ingredient116)
  (dispenses dispenser348 ingredient406)
  (dispenses dispenser82 ingredient122)
  (dispenses dispenser297 ingredient129)
  (clean shaker259)
  (clean shot306)
  (clean shot425)
  (clean shot488)
  (empty shaker259)
  (empty shot306)
  (empty shot425)
  (empty shot488)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail156 ingredient406)
  (cocktail-part2 cocktail156 ingredient122)
)
 (:goal
  (and
      (contains shot306 cocktail156)
      (contains shot425 ingredient129)
)))
