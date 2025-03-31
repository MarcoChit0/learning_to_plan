(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot55 shot186 - shot
      ingredient440 ingredient251 ingredient12 - ingredient
      cocktail375 - cocktail
      dispenser281 dispenser26 dispenser202 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot55)
  (ontable shot186)
  (dispenses dispenser281 ingredient440)
  (dispenses dispenser26 ingredient251)
  (dispenses dispenser202 ingredient12)
  (clean shaker385)
  (clean shot55)
  (clean shot186)
  (empty shaker385)
  (empty shot55)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient440)
  (cocktail-part2 cocktail375 ingredient12)
)
 (:goal
  (and
      (contains shot55 cocktail375)
)))
