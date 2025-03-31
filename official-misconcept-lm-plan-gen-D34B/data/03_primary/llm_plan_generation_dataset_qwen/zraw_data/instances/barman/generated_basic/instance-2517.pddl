(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot393 - shot
      ingredient391 ingredient95 - ingredient
      cocktail66 - cocktail
      dispenser158 dispenser180 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot393)
  (dispenses dispenser158 ingredient391)
  (dispenses dispenser180 ingredient95)
  (clean shaker400)
  (clean shot393)
  (empty shaker400)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail66 ingredient391)
  (cocktail-part2 cocktail66 ingredient95)
)
 (:goal
  (and
      (contains shot393 cocktail66)
)))
