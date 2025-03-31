(define (problem prob)
 (:domain barman)
 (:objects 
      shaker393 - shaker
      left right - hand
      shot411 shot31 - shot
      ingredient354 ingredient44 - ingredient
      cocktail1 - cocktail
      dispenser108 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker393)
  (ontable shot411)
  (ontable shot31)
  (dispenses dispenser108 ingredient354)
  (dispenses dispenser266 ingredient44)
  (clean shaker393)
  (clean shot411)
  (clean shot31)
  (empty shaker393)
  (empty shot411)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker393 l0)
  (shaker-level shaker393 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient44)
  (cocktail-part2 cocktail1 ingredient354)
)
 (:goal
  (and
      (contains shot411 cocktail1)
)))
