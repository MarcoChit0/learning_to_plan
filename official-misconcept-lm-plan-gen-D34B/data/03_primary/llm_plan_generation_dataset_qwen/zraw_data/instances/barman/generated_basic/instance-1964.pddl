(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot13 shot48 shot312 - shot
      ingredient178 ingredient364 ingredient164 ingredient363 - ingredient
      cocktail271 - cocktail
      dispenser305 dispenser452 dispenser408 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot13)
  (ontable shot48)
  (ontable shot312)
  (dispenses dispenser305 ingredient178)
  (dispenses dispenser452 ingredient364)
  (dispenses dispenser408 ingredient164)
  (dispenses dispenser365 ingredient363)
  (clean shaker221)
  (clean shot13)
  (clean shot48)
  (clean shot312)
  (empty shaker221)
  (empty shot13)
  (empty shot48)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient178)
  (cocktail-part2 cocktail271 ingredient363)
)
 (:goal
  (and
      (contains shot13 cocktail271)
      (contains shot48 cocktail271)
)))
