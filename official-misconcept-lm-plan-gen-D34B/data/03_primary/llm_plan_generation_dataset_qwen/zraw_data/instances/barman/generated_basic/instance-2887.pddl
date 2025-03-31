(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot323 shot391 shot394 - shot
      ingredient215 ingredient199 - ingredient
      cocktail206 - cocktail
      dispenser218 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot323)
  (ontable shot391)
  (ontable shot394)
  (dispenses dispenser218 ingredient215)
  (dispenses dispenser176 ingredient199)
  (clean shaker314)
  (clean shot323)
  (clean shot391)
  (clean shot394)
  (empty shaker314)
  (empty shot323)
  (empty shot391)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient215)
  (cocktail-part2 cocktail206 ingredient199)
)
 (:goal
  (and
      (contains shot323 cocktail206)
      (contains shot391 ingredient199)
)))
