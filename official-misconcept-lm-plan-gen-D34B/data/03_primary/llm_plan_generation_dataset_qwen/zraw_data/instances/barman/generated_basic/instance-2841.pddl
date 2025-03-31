(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot436 shot225 - shot
      ingredient324 ingredient5 ingredient94 ingredient449 - ingredient
      cocktail156 - cocktail
      dispenser410 dispenser421 dispenser449 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot436)
  (ontable shot225)
  (dispenses dispenser410 ingredient324)
  (dispenses dispenser421 ingredient5)
  (dispenses dispenser449 ingredient94)
  (dispenses dispenser161 ingredient449)
  (clean shaker200)
  (clean shot436)
  (clean shot225)
  (empty shaker200)
  (empty shot436)
  (empty shot225)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail156 ingredient94)
  (cocktail-part2 cocktail156 ingredient324)
)
 (:goal
  (and
      (contains shot436 cocktail156)
)))
