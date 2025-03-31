(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot464 - shot
      ingredient69 ingredient276 ingredient403 ingredient201 - ingredient
      cocktail111 - cocktail
      dispenser474 dispenser41 dispenser435 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot464)
  (dispenses dispenser474 ingredient69)
  (dispenses dispenser41 ingredient276)
  (dispenses dispenser435 ingredient403)
  (dispenses dispenser250 ingredient201)
  (clean shaker450)
  (clean shot464)
  (empty shaker450)
  (empty shot464)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail111 ingredient201)
  (cocktail-part2 cocktail111 ingredient403)
)
 (:goal
  (and
      (contains shot464 cocktail111)
)))
