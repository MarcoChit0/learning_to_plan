(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot330 shot6 - shot
      ingredient172 ingredient6 - ingredient
      cocktail303 - cocktail
      dispenser345 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot330)
  (ontable shot6)
  (dispenses dispenser345 ingredient172)
  (dispenses dispenser115 ingredient6)
  (clean shaker462)
  (clean shot330)
  (clean shot6)
  (empty shaker462)
  (empty shot330)
  (empty shot6)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail303 ingredient6)
  (cocktail-part2 cocktail303 ingredient172)
)
 (:goal
  (and
      (contains shot330 cocktail303)
)))
