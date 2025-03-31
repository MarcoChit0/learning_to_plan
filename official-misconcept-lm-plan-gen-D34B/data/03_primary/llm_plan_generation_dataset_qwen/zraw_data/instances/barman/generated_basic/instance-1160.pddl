(define (problem prob)
 (:domain barman)
 (:objects 
      shaker105 - shaker
      left right - hand
      shot447 shot105 - shot
      ingredient289 ingredient446 ingredient43 ingredient222 - ingredient
      cocktail1 - cocktail
      dispenser114 dispenser447 dispenser107 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker105)
  (ontable shot447)
  (ontable shot105)
  (dispenses dispenser114 ingredient289)
  (dispenses dispenser447 ingredient446)
  (dispenses dispenser107 ingredient43)
  (dispenses dispenser205 ingredient222)
  (clean shaker105)
  (clean shot447)
  (clean shot105)
  (empty shaker105)
  (empty shot447)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker105 l0)
  (shaker-level shaker105 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient289)
  (cocktail-part2 cocktail1 ingredient43)
)
 (:goal
  (and
      (contains shot447 cocktail1)
)))
