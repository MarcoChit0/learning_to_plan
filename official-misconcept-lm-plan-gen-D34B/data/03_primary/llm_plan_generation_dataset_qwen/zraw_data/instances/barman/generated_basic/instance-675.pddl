(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot287 shot282 - shot
      ingredient148 ingredient76 - ingredient
      cocktail1 - cocktail
      dispenser314 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot287)
  (ontable shot282)
  (dispenses dispenser314 ingredient148)
  (dispenses dispenser441 ingredient76)
  (clean shaker433)
  (clean shot287)
  (clean shot282)
  (empty shaker433)
  (empty shot287)
  (empty shot282)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient148)
  (cocktail-part2 cocktail1 ingredient76)
)
 (:goal
  (and
      (contains shot287 cocktail1)
)))
