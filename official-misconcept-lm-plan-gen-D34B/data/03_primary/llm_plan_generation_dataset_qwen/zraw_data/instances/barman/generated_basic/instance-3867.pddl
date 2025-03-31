(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot17 shot290 - shot
      ingredient142 ingredient99 ingredient322 - ingredient
      cocktail223 - cocktail
      dispenser385 dispenser106 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot17)
  (ontable shot290)
  (dispenses dispenser385 ingredient142)
  (dispenses dispenser106 ingredient99)
  (dispenses dispenser458 ingredient322)
  (clean shaker225)
  (clean shot17)
  (clean shot290)
  (empty shaker225)
  (empty shot17)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient99)
  (cocktail-part2 cocktail223 ingredient142)
)
 (:goal
  (and
      (contains shot17 cocktail223)
)))
