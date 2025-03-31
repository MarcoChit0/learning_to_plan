(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot82 shot35 shot250 - shot
      ingredient365 ingredient355 - ingredient
      cocktail67 - cocktail
      dispenser60 dispenser493 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot82)
  (ontable shot35)
  (ontable shot250)
  (dispenses dispenser60 ingredient365)
  (dispenses dispenser493 ingredient355)
  (clean shaker4)
  (clean shot82)
  (clean shot35)
  (clean shot250)
  (empty shaker4)
  (empty shot82)
  (empty shot35)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail67 ingredient365)
  (cocktail-part2 cocktail67 ingredient355)
)
 (:goal
  (and
      (contains shot82 cocktail67)
      (contains shot35 ingredient355)
)))
