(define (problem prob)
 (:domain barman)
 (:objects 
      shaker406 - shaker
      left right - hand
      shot414 - shot
      ingredient369 ingredient105 - ingredient
      cocktail107 - cocktail
      dispenser451 dispenser198 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker406)
  (ontable shot414)
  (dispenses dispenser451 ingredient369)
  (dispenses dispenser198 ingredient105)
  (clean shaker406)
  (clean shot414)
  (empty shaker406)
  (empty shot414)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker406 l0)
  (shaker-level shaker406 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail107 ingredient105)
  (cocktail-part2 cocktail107 ingredient369)
)
 (:goal
  (and
      (contains shot414 cocktail107)
)))
