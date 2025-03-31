(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot18 shot484 - shot
      ingredient4 ingredient434 ingredient16 - ingredient
      cocktail389 - cocktail
      dispenser131 dispenser60 dispenser242 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot18)
  (ontable shot484)
  (dispenses dispenser131 ingredient4)
  (dispenses dispenser60 ingredient434)
  (dispenses dispenser242 ingredient16)
  (clean shaker19)
  (clean shot18)
  (clean shot484)
  (empty shaker19)
  (empty shot18)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail389 ingredient16)
  (cocktail-part2 cocktail389 ingredient4)
)
 (:goal
  (and
      (contains shot18 cocktail389)
)))
