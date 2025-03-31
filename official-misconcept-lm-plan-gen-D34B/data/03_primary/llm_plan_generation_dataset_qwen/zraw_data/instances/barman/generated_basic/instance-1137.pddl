(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot359 shot339 - shot
      ingredient363 ingredient243 ingredient51 - ingredient
      cocktail1 - cocktail
      dispenser221 dispenser344 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot359)
  (ontable shot339)
  (dispenses dispenser221 ingredient363)
  (dispenses dispenser344 ingredient243)
  (dispenses dispenser126 ingredient51)
  (clean shaker416)
  (clean shot359)
  (clean shot339)
  (empty shaker416)
  (empty shot359)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient243)
  (cocktail-part2 cocktail1 ingredient51)
)
 (:goal
  (and
      (contains shot359 cocktail1)
)))
