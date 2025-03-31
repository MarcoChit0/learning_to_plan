(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot496 shot273 shot477 - shot
      ingredient23 ingredient212 ingredient55 ingredient230 - ingredient
      cocktail1 - cocktail
      dispenser315 dispenser87 dispenser453 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot496)
  (ontable shot273)
  (ontable shot477)
  (dispenses dispenser315 ingredient23)
  (dispenses dispenser87 ingredient212)
  (dispenses dispenser453 ingredient55)
  (dispenses dispenser92 ingredient230)
  (clean shaker202)
  (clean shot496)
  (clean shot273)
  (clean shot477)
  (empty shaker202)
  (empty shot496)
  (empty shot273)
  (empty shot477)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient230)
  (cocktail-part2 cocktail1 ingredient55)
)
 (:goal
  (and
      (contains shot496 cocktail1)
      (contains shot273 ingredient23)
)))
