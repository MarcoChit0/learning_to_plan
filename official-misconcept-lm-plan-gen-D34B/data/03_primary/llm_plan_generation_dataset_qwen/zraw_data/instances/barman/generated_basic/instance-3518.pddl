(define (problem prob)
 (:domain barman)
 (:objects 
      shaker483 - shaker
      left right - hand
      shot64 shot4 - shot
      ingredient151 ingredient46 ingredient357 ingredient92 - ingredient
      cocktail191 - cocktail
      dispenser141 dispenser416 dispenser12 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker483)
  (ontable shot64)
  (ontable shot4)
  (dispenses dispenser141 ingredient151)
  (dispenses dispenser416 ingredient46)
  (dispenses dispenser12 ingredient357)
  (dispenses dispenser252 ingredient92)
  (clean shaker483)
  (clean shot64)
  (clean shot4)
  (empty shaker483)
  (empty shot64)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker483 l0)
  (shaker-level shaker483 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail191 ingredient151)
  (cocktail-part2 cocktail191 ingredient92)
)
 (:goal
  (and
      (contains shot64 cocktail191)
)))
