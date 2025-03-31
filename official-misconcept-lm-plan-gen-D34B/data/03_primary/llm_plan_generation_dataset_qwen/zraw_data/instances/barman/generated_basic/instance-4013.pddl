(define (problem prob)
 (:domain barman)
 (:objects 
      shaker203 - shaker
      left right - hand
      shot291 shot183 - shot
      ingredient367 ingredient319 ingredient477 - ingredient
      cocktail214 - cocktail
      dispenser344 dispenser445 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker203)
  (ontable shot291)
  (ontable shot183)
  (dispenses dispenser344 ingredient367)
  (dispenses dispenser445 ingredient319)
  (dispenses dispenser377 ingredient477)
  (clean shaker203)
  (clean shot291)
  (clean shot183)
  (empty shaker203)
  (empty shot291)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker203 l0)
  (shaker-level shaker203 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail214 ingredient319)
  (cocktail-part2 cocktail214 ingredient477)
)
 (:goal
  (and
      (contains shot291 cocktail214)
)))
