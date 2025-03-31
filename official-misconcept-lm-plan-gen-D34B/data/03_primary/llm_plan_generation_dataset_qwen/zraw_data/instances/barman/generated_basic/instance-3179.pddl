(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot38 - shot
      ingredient139 ingredient470 ingredient222 - ingredient
      cocktail270 - cocktail
      dispenser444 dispenser54 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot38)
  (dispenses dispenser444 ingredient139)
  (dispenses dispenser54 ingredient470)
  (dispenses dispenser7 ingredient222)
  (clean shaker170)
  (clean shot38)
  (empty shaker170)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail270 ingredient470)
  (cocktail-part2 cocktail270 ingredient139)
)
 (:goal
  (and
      (contains shot38 cocktail270)
)))
