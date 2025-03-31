(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot296 shot349 - shot
      ingredient279 ingredient440 ingredient471 - ingredient
      cocktail190 - cocktail
      dispenser29 dispenser431 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot296)
  (ontable shot349)
  (dispenses dispenser29 ingredient279)
  (dispenses dispenser431 ingredient440)
  (dispenses dispenser370 ingredient471)
  (clean shaker350)
  (clean shot296)
  (clean shot349)
  (empty shaker350)
  (empty shot296)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient471)
  (cocktail-part2 cocktail190 ingredient440)
)
 (:goal
  (and
      (contains shot296 cocktail190)
)))
