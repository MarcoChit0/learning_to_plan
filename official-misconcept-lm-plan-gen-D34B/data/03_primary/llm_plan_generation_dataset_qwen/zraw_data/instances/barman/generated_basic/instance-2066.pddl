(define (problem prob)
 (:domain barman)
 (:objects 
      shaker252 - shaker
      left right - hand
      shot245 shot55 - shot
      ingredient85 ingredient164 - ingredient
      cocktail57 - cocktail
      dispenser355 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker252)
  (ontable shot245)
  (ontable shot55)
  (dispenses dispenser355 ingredient85)
  (dispenses dispenser364 ingredient164)
  (clean shaker252)
  (clean shot245)
  (clean shot55)
  (empty shaker252)
  (empty shot245)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker252 l0)
  (shaker-level shaker252 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail57 ingredient164)
  (cocktail-part2 cocktail57 ingredient85)
)
 (:goal
  (and
      (contains shot245 cocktail57)
)))
