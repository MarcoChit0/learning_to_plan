(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot146 - shot
      ingredient142 ingredient260 - ingredient
      cocktail11 - cocktail
      dispenser487 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot146)
  (dispenses dispenser487 ingredient142)
  (dispenses dispenser200 ingredient260)
  (clean shaker162)
  (clean shot146)
  (empty shaker162)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail11 ingredient260)
  (cocktail-part2 cocktail11 ingredient142)
)
 (:goal
  (and
      (contains shot146 cocktail11)
)))
