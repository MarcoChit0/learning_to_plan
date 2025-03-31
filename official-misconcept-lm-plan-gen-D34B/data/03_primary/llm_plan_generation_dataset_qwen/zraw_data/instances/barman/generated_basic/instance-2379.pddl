(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot252 - shot
      ingredient70 ingredient46 ingredient361 ingredient82 - ingredient
      cocktail213 - cocktail
      dispenser356 dispenser179 dispenser23 dispenser18 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot252)
  (dispenses dispenser356 ingredient70)
  (dispenses dispenser179 ingredient46)
  (dispenses dispenser23 ingredient361)
  (dispenses dispenser18 ingredient82)
  (clean shaker489)
  (clean shot252)
  (empty shaker489)
  (empty shot252)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail213 ingredient82)
  (cocktail-part2 cocktail213 ingredient70)
)
 (:goal
  (and
      (contains shot252 cocktail213)
)))
