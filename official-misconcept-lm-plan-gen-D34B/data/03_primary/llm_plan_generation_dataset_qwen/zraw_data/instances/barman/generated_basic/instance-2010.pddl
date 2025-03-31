(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot409 shot47 shot470 - shot
      ingredient67 ingredient63 ingredient183 - ingredient
      cocktail46 - cocktail
      dispenser100 dispenser155 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot409)
  (ontable shot47)
  (ontable shot470)
  (dispenses dispenser100 ingredient67)
  (dispenses dispenser155 ingredient63)
  (dispenses dispenser152 ingredient183)
  (clean shaker360)
  (clean shot409)
  (clean shot47)
  (clean shot470)
  (empty shaker360)
  (empty shot409)
  (empty shot47)
  (empty shot470)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail46 ingredient63)
  (cocktail-part2 cocktail46 ingredient67)
)
 (:goal
  (and
      (contains shot409 cocktail46)
      (contains shot47 cocktail46)
)))
