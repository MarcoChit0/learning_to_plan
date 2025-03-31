(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot390 shot65 shot319 - shot
      ingredient149 ingredient38 ingredient290 - ingredient
      cocktail1 - cocktail
      dispenser250 dispenser378 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot390)
  (ontable shot65)
  (ontable shot319)
  (dispenses dispenser250 ingredient149)
  (dispenses dispenser378 ingredient38)
  (dispenses dispenser368 ingredient290)
  (clean shaker330)
  (clean shot390)
  (clean shot65)
  (clean shot319)
  (empty shaker330)
  (empty shot390)
  (empty shot65)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient290)
  (cocktail-part2 cocktail1 ingredient149)
)
 (:goal
  (and
      (contains shot390 cocktail1)
      (contains shot65 cocktail1)
)))
