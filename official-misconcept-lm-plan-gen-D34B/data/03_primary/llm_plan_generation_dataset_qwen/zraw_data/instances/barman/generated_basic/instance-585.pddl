(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot96 shot392 - shot
      ingredient105 ingredient196 - ingredient
      cocktail1 - cocktail
      dispenser123 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot96)
  (ontable shot392)
  (dispenses dispenser123 ingredient105)
  (dispenses dispenser385 ingredient196)
  (clean shaker188)
  (clean shot96)
  (clean shot392)
  (empty shaker188)
  (empty shot96)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient196)
  (cocktail-part2 cocktail1 ingredient105)
)
 (:goal
  (and
      (contains shot96 cocktail1)
)))
