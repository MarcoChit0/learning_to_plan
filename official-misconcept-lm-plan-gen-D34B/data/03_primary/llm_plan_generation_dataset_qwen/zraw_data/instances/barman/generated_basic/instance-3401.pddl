(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot477 shot449 - shot
      ingredient348 ingredient136 ingredient447 ingredient161 - ingredient
      cocktail473 - cocktail
      dispenser141 dispenser118 dispenser356 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot477)
  (ontable shot449)
  (dispenses dispenser141 ingredient348)
  (dispenses dispenser118 ingredient136)
  (dispenses dispenser356 ingredient447)
  (dispenses dispenser161 ingredient161)
  (clean shaker463)
  (clean shot477)
  (clean shot449)
  (empty shaker463)
  (empty shot477)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient136)
  (cocktail-part2 cocktail473 ingredient161)
)
 (:goal
  (and
      (contains shot477 cocktail473)
)))
