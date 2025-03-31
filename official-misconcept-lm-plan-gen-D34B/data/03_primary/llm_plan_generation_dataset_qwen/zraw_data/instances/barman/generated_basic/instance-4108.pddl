(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot167 shot395 - shot
      ingredient298 ingredient413 - ingredient
      cocktail87 - cocktail
      dispenser495 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot167)
  (ontable shot395)
  (dispenses dispenser495 ingredient298)
  (dispenses dispenser49 ingredient413)
  (clean shaker180)
  (clean shot167)
  (clean shot395)
  (empty shaker180)
  (empty shot167)
  (empty shot395)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient413)
  (cocktail-part2 cocktail87 ingredient298)
)
 (:goal
  (and
      (contains shot167 cocktail87)
)))
