(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot87 shot201 - shot
      ingredient212 ingredient228 ingredient415 ingredient242 - ingredient
      cocktail241 - cocktail
      dispenser99 dispenser441 dispenser286 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot87)
  (ontable shot201)
  (dispenses dispenser99 ingredient212)
  (dispenses dispenser441 ingredient228)
  (dispenses dispenser286 ingredient415)
  (dispenses dispenser211 ingredient242)
  (clean shaker38)
  (clean shot87)
  (clean shot201)
  (empty shaker38)
  (empty shot87)
  (empty shot201)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient415)
  (cocktail-part2 cocktail241 ingredient212)
)
 (:goal
  (and
      (contains shot87 cocktail241)
)))
