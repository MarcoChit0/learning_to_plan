(define (problem prob)
 (:domain barman)
 (:objects 
      shaker112 - shaker
      left right - hand
      shot438 - shot
      ingredient273 ingredient74 ingredient52 ingredient491 - ingredient
      cocktail347 - cocktail
      dispenser96 dispenser347 dispenser245 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker112)
  (ontable shot438)
  (dispenses dispenser96 ingredient273)
  (dispenses dispenser347 ingredient74)
  (dispenses dispenser245 ingredient52)
  (dispenses dispenser106 ingredient491)
  (clean shaker112)
  (clean shot438)
  (empty shaker112)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker112 l0)
  (shaker-level shaker112 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail347 ingredient491)
  (cocktail-part2 cocktail347 ingredient52)
)
 (:goal
  (and
      (contains shot438 cocktail347)
)))
