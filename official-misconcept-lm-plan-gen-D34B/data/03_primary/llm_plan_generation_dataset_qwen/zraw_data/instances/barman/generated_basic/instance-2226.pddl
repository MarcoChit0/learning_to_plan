(define (problem prob)
 (:domain barman)
 (:objects 
      shaker249 - shaker
      left right - hand
      shot268 - shot
      ingredient389 ingredient279 - ingredient
      cocktail95 - cocktail
      dispenser255 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker249)
  (ontable shot268)
  (dispenses dispenser255 ingredient389)
  (dispenses dispenser73 ingredient279)
  (clean shaker249)
  (clean shot268)
  (empty shaker249)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker249 l0)
  (shaker-level shaker249 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail95 ingredient389)
  (cocktail-part2 cocktail95 ingredient279)
)
 (:goal
  (and
      (contains shot268 cocktail95)
)))
