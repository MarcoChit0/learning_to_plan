(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot483 shot36 - shot
      ingredient231 ingredient19 ingredient440 - ingredient
      cocktail395 - cocktail
      dispenser268 dispenser181 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot483)
  (ontable shot36)
  (dispenses dispenser268 ingredient231)
  (dispenses dispenser181 ingredient19)
  (dispenses dispenser385 ingredient440)
  (clean shaker461)
  (clean shot483)
  (clean shot36)
  (empty shaker461)
  (empty shot483)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail395 ingredient19)
  (cocktail-part2 cocktail395 ingredient231)
)
 (:goal
  (and
      (contains shot483 cocktail395)
)))
