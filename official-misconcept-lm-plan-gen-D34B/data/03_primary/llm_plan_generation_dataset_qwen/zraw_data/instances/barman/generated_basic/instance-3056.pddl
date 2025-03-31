(define (problem prob)
 (:domain barman)
 (:objects 
      shaker92 - shaker
      left right - hand
      shot347 shot461 - shot
      ingredient467 ingredient405 ingredient84 - ingredient
      cocktail214 - cocktail
      dispenser219 dispenser294 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker92)
  (ontable shot347)
  (ontable shot461)
  (dispenses dispenser219 ingredient467)
  (dispenses dispenser294 ingredient405)
  (dispenses dispenser385 ingredient84)
  (clean shaker92)
  (clean shot347)
  (clean shot461)
  (empty shaker92)
  (empty shot347)
  (empty shot461)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker92 l0)
  (shaker-level shaker92 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail214 ingredient405)
  (cocktail-part2 cocktail214 ingredient84)
)
 (:goal
  (and
      (contains shot347 cocktail214)
)))
