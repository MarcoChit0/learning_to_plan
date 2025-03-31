(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot289 shot215 shot208 - shot
      ingredient434 ingredient297 ingredient299 ingredient52 - ingredient
      cocktail192 - cocktail
      dispenser54 dispenser413 dispenser162 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot289)
  (ontable shot215)
  (ontable shot208)
  (dispenses dispenser54 ingredient434)
  (dispenses dispenser413 ingredient297)
  (dispenses dispenser162 ingredient299)
  (dispenses dispenser165 ingredient52)
  (clean shaker39)
  (clean shot289)
  (clean shot215)
  (clean shot208)
  (empty shaker39)
  (empty shot289)
  (empty shot215)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail192 ingredient297)
  (cocktail-part2 cocktail192 ingredient434)
)
 (:goal
  (and
      (contains shot289 cocktail192)
      (contains shot215 ingredient434)
)))
