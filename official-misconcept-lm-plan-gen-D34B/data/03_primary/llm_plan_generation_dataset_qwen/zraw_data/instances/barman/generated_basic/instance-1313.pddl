(define (problem prob)
 (:domain barman)
 (:objects 
      shaker422 - shaker
      left right - hand
      shot295 shot415 shot99 shot202 - shot
      ingredient364 ingredient212 - ingredient
      cocktail400 - cocktail
      dispenser148 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker422)
  (ontable shot295)
  (ontable shot415)
  (ontable shot99)
  (ontable shot202)
  (dispenses dispenser148 ingredient364)
  (dispenses dispenser369 ingredient212)
  (clean shaker422)
  (clean shot295)
  (clean shot415)
  (clean shot99)
  (clean shot202)
  (empty shaker422)
  (empty shot295)
  (empty shot415)
  (empty shot99)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker422 l0)
  (shaker-level shaker422 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail400 ingredient364)
  (cocktail-part2 cocktail400 ingredient212)
)
 (:goal
  (and
      (contains shot295 cocktail400)
      (contains shot415 ingredient212)
      (contains shot99 cocktail400)
)))
