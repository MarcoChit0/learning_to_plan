(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot251 - shot
      ingredient18 ingredient178 ingredient256 ingredient354 - ingredient
      cocktail107 - cocktail
      dispenser428 dispenser301 dispenser198 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot251)
  (dispenses dispenser428 ingredient18)
  (dispenses dispenser301 ingredient178)
  (dispenses dispenser198 ingredient256)
  (dispenses dispenser165 ingredient354)
  (clean shaker190)
  (clean shot251)
  (empty shaker190)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail107 ingredient178)
  (cocktail-part2 cocktail107 ingredient354)
)
 (:goal
  (and
      (contains shot251 cocktail107)
)))
