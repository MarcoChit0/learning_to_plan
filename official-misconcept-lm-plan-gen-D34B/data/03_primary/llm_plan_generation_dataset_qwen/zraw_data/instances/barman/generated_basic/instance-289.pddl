(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot107 shot25 shot236 - shot
      ingredient228 ingredient406 ingredient26 ingredient155 - ingredient
      cocktail1 - cocktail
      dispenser30 dispenser468 dispenser425 dispenser443 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot107)
  (ontable shot25)
  (ontable shot236)
  (dispenses dispenser30 ingredient228)
  (dispenses dispenser468 ingredient406)
  (dispenses dispenser425 ingredient26)
  (dispenses dispenser443 ingredient155)
  (clean shaker341)
  (clean shot107)
  (clean shot25)
  (clean shot236)
  (empty shaker341)
  (empty shot107)
  (empty shot25)
  (empty shot236)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient228)
  (cocktail-part2 cocktail1 ingredient406)
)
 (:goal
  (and
      (contains shot107 cocktail1)
      (contains shot25 cocktail1)
)))
