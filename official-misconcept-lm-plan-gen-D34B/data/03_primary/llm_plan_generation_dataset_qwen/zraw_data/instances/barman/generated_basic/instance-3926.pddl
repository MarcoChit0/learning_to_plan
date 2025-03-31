(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot420 - shot
      ingredient126 ingredient288 ingredient162 ingredient110 - ingredient
      cocktail204 - cocktail
      dispenser209 dispenser406 dispenser435 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot420)
  (dispenses dispenser209 ingredient126)
  (dispenses dispenser406 ingredient288)
  (dispenses dispenser435 ingredient162)
  (dispenses dispenser74 ingredient110)
  (clean shaker298)
  (clean shot420)
  (empty shaker298)
  (empty shot420)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail204 ingredient126)
  (cocktail-part2 cocktail204 ingredient162)
)
 (:goal
  (and
      (contains shot420 cocktail204)
)))
