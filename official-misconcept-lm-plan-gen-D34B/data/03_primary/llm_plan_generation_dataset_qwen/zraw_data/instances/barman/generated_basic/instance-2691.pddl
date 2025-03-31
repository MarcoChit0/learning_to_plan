(define (problem prob)
 (:domain barman)
 (:objects 
      shaker105 - shaker
      left right - hand
      shot308 shot298 - shot
      ingredient365 ingredient276 ingredient403 - ingredient
      cocktail141 - cocktail
      dispenser138 dispenser431 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker105)
  (ontable shot308)
  (ontable shot298)
  (dispenses dispenser138 ingredient365)
  (dispenses dispenser431 ingredient276)
  (dispenses dispenser196 ingredient403)
  (clean shaker105)
  (clean shot308)
  (clean shot298)
  (empty shaker105)
  (empty shot308)
  (empty shot298)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker105 l0)
  (shaker-level shaker105 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient365)
  (cocktail-part2 cocktail141 ingredient403)
)
 (:goal
  (and
      (contains shot308 cocktail141)
)))
