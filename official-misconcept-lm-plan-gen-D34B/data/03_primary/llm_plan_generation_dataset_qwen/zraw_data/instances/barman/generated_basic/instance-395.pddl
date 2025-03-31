(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot423 shot237 - shot
      ingredient454 ingredient220 - ingredient
      cocktail1 - cocktail
      dispenser110 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot423)
  (ontable shot237)
  (dispenses dispenser110 ingredient454)
  (dispenses dispenser467 ingredient220)
  (clean shaker487)
  (clean shot423)
  (clean shot237)
  (empty shaker487)
  (empty shot423)
  (empty shot237)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient220)
  (cocktail-part2 cocktail1 ingredient454)
)
 (:goal
  (and
      (contains shot423 cocktail1)
)))
