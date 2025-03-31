(define (problem prob)
 (:domain barman)
 (:objects 
      shaker456 - shaker
      left right - hand
      shot172 shot190 - shot
      ingredient231 ingredient304 - ingredient
      cocktail1 - cocktail
      dispenser138 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker456)
  (ontable shot172)
  (ontable shot190)
  (dispenses dispenser138 ingredient231)
  (dispenses dispenser428 ingredient304)
  (clean shaker456)
  (clean shot172)
  (clean shot190)
  (empty shaker456)
  (empty shot172)
  (empty shot190)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker456 l0)
  (shaker-level shaker456 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient304)
  (cocktail-part2 cocktail1 ingredient231)
)
 (:goal
  (and
      (contains shot172 cocktail1)
)))
