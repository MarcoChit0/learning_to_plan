(define (problem prob)
 (:domain barman)
 (:objects 
      shaker480 - shaker
      left right - hand
      shot463 shot49 - shot
      ingredient9 ingredient318 ingredient376 - ingredient
      cocktail1 - cocktail
      dispenser178 dispenser333 dispenser173 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker480)
  (ontable shot463)
  (ontable shot49)
  (dispenses dispenser178 ingredient9)
  (dispenses dispenser333 ingredient318)
  (dispenses dispenser173 ingredient376)
  (clean shaker480)
  (clean shot463)
  (clean shot49)
  (empty shaker480)
  (empty shot463)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker480 l0)
  (shaker-level shaker480 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient376)
  (cocktail-part2 cocktail1 ingredient9)
)
 (:goal
  (and
      (contains shot463 cocktail1)
)))
