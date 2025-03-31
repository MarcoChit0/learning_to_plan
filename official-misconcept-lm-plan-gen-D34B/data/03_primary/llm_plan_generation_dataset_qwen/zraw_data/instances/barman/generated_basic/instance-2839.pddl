(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot59 shot171 - shot
      ingredient410 ingredient464 ingredient333 ingredient433 - ingredient
      cocktail365 - cocktail
      dispenser45 dispenser422 dispenser238 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot59)
  (ontable shot171)
  (dispenses dispenser45 ingredient410)
  (dispenses dispenser422 ingredient464)
  (dispenses dispenser238 ingredient333)
  (dispenses dispenser157 ingredient433)
  (clean shaker212)
  (clean shot59)
  (clean shot171)
  (empty shaker212)
  (empty shot59)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient333)
  (cocktail-part2 cocktail365 ingredient433)
)
 (:goal
  (and
      (contains shot59 cocktail365)
)))
