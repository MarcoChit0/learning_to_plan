(define (problem prob)
 (:domain barman)
 (:objects 
      shaker323 - shaker
      left right - hand
      shot228 shot358 - shot
      ingredient270 ingredient6 ingredient178 - ingredient
      cocktail43 - cocktail
      dispenser254 dispenser103 dispenser121 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker323)
  (ontable shot228)
  (ontable shot358)
  (dispenses dispenser254 ingredient270)
  (dispenses dispenser103 ingredient6)
  (dispenses dispenser121 ingredient178)
  (clean shaker323)
  (clean shot228)
  (clean shot358)
  (empty shaker323)
  (empty shot228)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker323 l0)
  (shaker-level shaker323 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail43 ingredient178)
  (cocktail-part2 cocktail43 ingredient270)
)
 (:goal
  (and
      (contains shot228 cocktail43)
)))
