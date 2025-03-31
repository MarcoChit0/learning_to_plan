(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot197 shot458 shot230 - shot
      ingredient212 ingredient355 ingredient27 ingredient494 - ingredient
      cocktail402 - cocktail
      dispenser177 dispenser86 dispenser459 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot197)
  (ontable shot458)
  (ontable shot230)
  (dispenses dispenser177 ingredient212)
  (dispenses dispenser86 ingredient355)
  (dispenses dispenser459 ingredient27)
  (dispenses dispenser322 ingredient494)
  (clean shaker426)
  (clean shot197)
  (clean shot458)
  (clean shot230)
  (empty shaker426)
  (empty shot197)
  (empty shot458)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail402 ingredient212)
  (cocktail-part2 cocktail402 ingredient27)
)
 (:goal
  (and
      (contains shot197 cocktail402)
      (contains shot458 cocktail402)
)))
