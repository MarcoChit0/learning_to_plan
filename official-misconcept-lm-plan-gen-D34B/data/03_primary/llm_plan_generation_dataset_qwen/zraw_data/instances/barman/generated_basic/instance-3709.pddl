(define (problem prob)
 (:domain barman)
 (:objects 
      shaker214 - shaker
      left right - hand
      shot452 shot198 shot89 shot82 - shot
      ingredient160 ingredient193 - ingredient
      cocktail395 - cocktail
      dispenser401 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker214)
  (ontable shot452)
  (ontable shot198)
  (ontable shot89)
  (ontable shot82)
  (dispenses dispenser401 ingredient160)
  (dispenses dispenser479 ingredient193)
  (clean shaker214)
  (clean shot452)
  (clean shot198)
  (clean shot89)
  (clean shot82)
  (empty shaker214)
  (empty shot452)
  (empty shot198)
  (empty shot89)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker214 l0)
  (shaker-level shaker214 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail395 ingredient193)
  (cocktail-part2 cocktail395 ingredient160)
)
 (:goal
  (and
      (contains shot452 cocktail395)
      (contains shot198 ingredient193)
      (contains shot89 cocktail395)
)))
