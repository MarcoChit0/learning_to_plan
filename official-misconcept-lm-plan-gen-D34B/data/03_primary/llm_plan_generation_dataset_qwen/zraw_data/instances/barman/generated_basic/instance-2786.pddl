(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot415 - shot
      ingredient283 ingredient178 ingredient100 ingredient69 - ingredient
      cocktail113 - cocktail
      dispenser163 dispenser351 dispenser61 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot415)
  (dispenses dispenser163 ingredient283)
  (dispenses dispenser351 ingredient178)
  (dispenses dispenser61 ingredient100)
  (dispenses dispenser109 ingredient69)
  (clean shaker139)
  (clean shot415)
  (empty shaker139)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail113 ingredient69)
  (cocktail-part2 cocktail113 ingredient178)
)
 (:goal
  (and
      (contains shot415 cocktail113)
)))
