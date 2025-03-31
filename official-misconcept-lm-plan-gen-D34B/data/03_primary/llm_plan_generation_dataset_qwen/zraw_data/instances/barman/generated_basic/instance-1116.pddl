(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot89 shot123 - shot
      ingredient425 ingredient400 - ingredient
      cocktail1 - cocktail
      dispenser457 dispenser168 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot89)
  (ontable shot123)
  (dispenses dispenser457 ingredient425)
  (dispenses dispenser168 ingredient400)
  (clean shaker15)
  (clean shot89)
  (clean shot123)
  (empty shaker15)
  (empty shot89)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient425)
  (cocktail-part2 cocktail1 ingredient400)
)
 (:goal
  (and
      (contains shot89 cocktail1)
)))
