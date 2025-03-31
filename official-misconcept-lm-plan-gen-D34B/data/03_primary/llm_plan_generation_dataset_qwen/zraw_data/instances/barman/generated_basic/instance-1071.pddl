(define (problem prob)
 (:domain barman)
 (:objects 
      shaker248 - shaker
      left right - hand
      shot425 shot222 shot493 - shot
      ingredient470 ingredient170 ingredient483 ingredient254 - ingredient
      cocktail1 - cocktail
      dispenser357 dispenser92 dispenser247 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker248)
  (ontable shot425)
  (ontable shot222)
  (ontable shot493)
  (dispenses dispenser357 ingredient470)
  (dispenses dispenser92 ingredient170)
  (dispenses dispenser247 ingredient483)
  (dispenses dispenser290 ingredient254)
  (clean shaker248)
  (clean shot425)
  (clean shot222)
  (clean shot493)
  (empty shaker248)
  (empty shot425)
  (empty shot222)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker248 l0)
  (shaker-level shaker248 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient254)
  (cocktail-part2 cocktail1 ingredient470)
)
 (:goal
  (and
      (contains shot425 cocktail1)
      (contains shot222 cocktail1)
)))
