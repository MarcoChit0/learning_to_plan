(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot21 shot247 shot431 - shot
      ingredient201 ingredient0 ingredient315 - ingredient
      cocktail73 - cocktail
      dispenser7 dispenser38 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot21)
  (ontable shot247)
  (ontable shot431)
  (dispenses dispenser7 ingredient201)
  (dispenses dispenser38 ingredient0)
  (dispenses dispenser305 ingredient315)
  (clean shaker341)
  (clean shot21)
  (clean shot247)
  (clean shot431)
  (empty shaker341)
  (empty shot21)
  (empty shot247)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail73 ingredient315)
  (cocktail-part2 cocktail73 ingredient201)
)
 (:goal
  (and
      (contains shot21 cocktail73)
      (contains shot247 cocktail73)
)))
