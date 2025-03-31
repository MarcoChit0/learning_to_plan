(define (problem prob)
 (:domain barman)
 (:objects 
      shaker243 - shaker
      left right - hand
      shot434 shot400 - shot
      ingredient414 ingredient438 ingredient255 ingredient136 - ingredient
      cocktail446 - cocktail
      dispenser137 dispenser421 dispenser359 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker243)
  (ontable shot434)
  (ontable shot400)
  (dispenses dispenser137 ingredient414)
  (dispenses dispenser421 ingredient438)
  (dispenses dispenser359 ingredient255)
  (dispenses dispenser151 ingredient136)
  (clean shaker243)
  (clean shot434)
  (clean shot400)
  (empty shaker243)
  (empty shot434)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker243 l0)
  (shaker-level shaker243 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail446 ingredient255)
  (cocktail-part2 cocktail446 ingredient136)
)
 (:goal
  (and
      (contains shot434 cocktail446)
)))
