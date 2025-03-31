(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot49 shot27 - shot
      ingredient239 ingredient45 ingredient318 ingredient219 - ingredient
      cocktail1 - cocktail
      dispenser360 dispenser404 dispenser54 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot49)
  (ontable shot27)
  (dispenses dispenser360 ingredient239)
  (dispenses dispenser404 ingredient45)
  (dispenses dispenser54 ingredient318)
  (dispenses dispenser403 ingredient219)
  (clean shaker244)
  (clean shot49)
  (clean shot27)
  (empty shaker244)
  (empty shot49)
  (empty shot27)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient45)
  (cocktail-part2 cocktail1 ingredient239)
)
 (:goal
  (and
      (contains shot49 cocktail1)
)))
