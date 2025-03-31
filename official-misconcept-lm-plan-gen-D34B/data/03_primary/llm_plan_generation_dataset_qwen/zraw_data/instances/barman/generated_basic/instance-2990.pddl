(define (problem prob)
 (:domain barman)
 (:objects 
      shaker496 - shaker
      left right - hand
      shot155 shot220 - shot
      ingredient104 ingredient400 ingredient491 ingredient157 - ingredient
      cocktail153 - cocktail
      dispenser353 dispenser12 dispenser218 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker496)
  (ontable shot155)
  (ontable shot220)
  (dispenses dispenser353 ingredient104)
  (dispenses dispenser12 ingredient400)
  (dispenses dispenser218 ingredient491)
  (dispenses dispenser427 ingredient157)
  (clean shaker496)
  (clean shot155)
  (clean shot220)
  (empty shaker496)
  (empty shot155)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker496 l0)
  (shaker-level shaker496 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail153 ingredient400)
  (cocktail-part2 cocktail153 ingredient157)
)
 (:goal
  (and
      (contains shot155 cocktail153)
)))
