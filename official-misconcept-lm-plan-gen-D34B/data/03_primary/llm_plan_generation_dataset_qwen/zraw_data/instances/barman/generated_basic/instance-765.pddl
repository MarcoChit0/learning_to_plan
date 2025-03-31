(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot198 - shot
      ingredient278 ingredient413 - ingredient
      cocktail1 - cocktail
      dispenser423 dispenser42 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot198)
  (dispenses dispenser423 ingredient278)
  (dispenses dispenser42 ingredient413)
  (clean shaker188)
  (clean shot198)
  (empty shaker188)
  (empty shot198)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient278)
  (cocktail-part2 cocktail1 ingredient413)
)
 (:goal
  (and
      (contains shot198 cocktail1)
)))
