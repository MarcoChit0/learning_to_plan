(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot139 - shot
      ingredient331 ingredient482 ingredient290 ingredient296 - ingredient
      cocktail240 - cocktail
      dispenser43 dispenser250 dispenser485 dispenser255 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot139)
  (dispenses dispenser43 ingredient331)
  (dispenses dispenser250 ingredient482)
  (dispenses dispenser485 ingredient290)
  (dispenses dispenser255 ingredient296)
  (clean shaker384)
  (clean shot139)
  (empty shaker384)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail240 ingredient290)
  (cocktail-part2 cocktail240 ingredient331)
)
 (:goal
  (and
      (contains shot139 cocktail240)
)))
