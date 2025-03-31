(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot118 shot212 shot232 - shot
      ingredient425 ingredient357 ingredient458 ingredient228 - ingredient
      cocktail1 - cocktail
      dispenser444 dispenser142 dispenser419 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot118)
  (ontable shot212)
  (ontable shot232)
  (dispenses dispenser444 ingredient425)
  (dispenses dispenser142 ingredient357)
  (dispenses dispenser419 ingredient458)
  (dispenses dispenser412 ingredient228)
  (clean shaker412)
  (clean shot118)
  (clean shot212)
  (clean shot232)
  (empty shaker412)
  (empty shot118)
  (empty shot212)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient228)
  (cocktail-part2 cocktail1 ingredient357)
)
 (:goal
  (and
      (contains shot118 cocktail1)
      (contains shot212 ingredient425)
)))
