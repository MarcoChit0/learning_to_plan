(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot420 shot479 - shot
      ingredient481 ingredient379 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser184 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot420)
  (ontable shot479)
  (dispenses dispenser184 ingredient481)
  (dispenses dispenser156 ingredient379)
  (clean shaker312)
  (clean shot420)
  (clean shot479)
  (empty shaker312)
  (empty shot420)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient379)
  (cocktail-part2 cocktail1 ingredient481)
  (cocktail-part1 cocktail2 ingredient481)
  (cocktail-part2 cocktail2 ingredient379)
)
 (:goal
  (and
      (contains shot420 cocktail1)
      (contains shot479 cocktail2)
)))
