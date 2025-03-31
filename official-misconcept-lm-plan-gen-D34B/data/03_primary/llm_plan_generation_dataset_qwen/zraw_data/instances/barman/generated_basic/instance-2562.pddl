(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot124 shot252 shot63 - shot
      ingredient104 ingredient398 - ingredient
      cocktail330 - cocktail
      dispenser395 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot124)
  (ontable shot252)
  (ontable shot63)
  (dispenses dispenser395 ingredient104)
  (dispenses dispenser224 ingredient398)
  (clean shaker350)
  (clean shot124)
  (clean shot252)
  (clean shot63)
  (empty shaker350)
  (empty shot124)
  (empty shot252)
  (empty shot63)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient104)
  (cocktail-part2 cocktail330 ingredient398)
)
 (:goal
  (and
      (contains shot124 cocktail330)
      (contains shot252 ingredient398)
)))
