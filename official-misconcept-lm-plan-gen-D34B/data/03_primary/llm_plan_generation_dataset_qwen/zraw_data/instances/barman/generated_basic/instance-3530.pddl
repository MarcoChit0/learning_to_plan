(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot358 shot26 - shot
      ingredient265 ingredient222 ingredient363 ingredient373 - ingredient
      cocktail404 - cocktail
      dispenser438 dispenser337 dispenser230 dispenser496 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot358)
  (ontable shot26)
  (dispenses dispenser438 ingredient265)
  (dispenses dispenser337 ingredient222)
  (dispenses dispenser230 ingredient363)
  (dispenses dispenser496 ingredient373)
  (clean shaker478)
  (clean shot358)
  (clean shot26)
  (empty shaker478)
  (empty shot358)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail404 ingredient363)
  (cocktail-part2 cocktail404 ingredient265)
)
 (:goal
  (and
      (contains shot358 cocktail404)
)))
