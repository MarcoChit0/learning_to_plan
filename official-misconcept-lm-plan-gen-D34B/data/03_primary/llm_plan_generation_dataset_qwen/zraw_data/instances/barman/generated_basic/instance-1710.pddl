(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot212 - shot
      ingredient316 ingredient37 ingredient28 ingredient356 - ingredient
      cocktail169 - cocktail
      dispenser354 dispenser454 dispenser358 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot212)
  (dispenses dispenser354 ingredient316)
  (dispenses dispenser454 ingredient37)
  (dispenses dispenser358 ingredient28)
  (dispenses dispenser100 ingredient356)
  (clean shaker225)
  (clean shot212)
  (empty shaker225)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail169 ingredient37)
  (cocktail-part2 cocktail169 ingredient356)
)
 (:goal
  (and
      (contains shot212 cocktail169)
)))
