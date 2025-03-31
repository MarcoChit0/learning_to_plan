(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot266 shot59 - shot
      ingredient460 ingredient174 ingredient395 ingredient411 - ingredient
      cocktail183 - cocktail
      dispenser298 dispenser322 dispenser339 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot266)
  (ontable shot59)
  (dispenses dispenser298 ingredient460)
  (dispenses dispenser322 ingredient174)
  (dispenses dispenser339 ingredient395)
  (dispenses dispenser141 ingredient411)
  (clean shaker99)
  (clean shot266)
  (clean shot59)
  (empty shaker99)
  (empty shot266)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient411)
  (cocktail-part2 cocktail183 ingredient395)
)
 (:goal
  (and
      (contains shot266 cocktail183)
)))
