(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot103 shot393 shot345 - shot
      ingredient51 ingredient178 - ingredient
      cocktail15 - cocktail
      dispenser269 dispenser448 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot103)
  (ontable shot393)
  (ontable shot345)
  (dispenses dispenser269 ingredient51)
  (dispenses dispenser448 ingredient178)
  (clean shaker34)
  (clean shot103)
  (clean shot393)
  (clean shot345)
  (empty shaker34)
  (empty shot103)
  (empty shot393)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail15 ingredient51)
  (cocktail-part2 cocktail15 ingredient178)
)
 (:goal
  (and
      (contains shot103 cocktail15)
      (contains shot393 cocktail15)
)))
