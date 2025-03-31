(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot414 shot365 - shot
      ingredient464 ingredient99 ingredient301 - ingredient
      cocktail1 - cocktail
      dispenser375 dispenser17 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot414)
  (ontable shot365)
  (dispenses dispenser375 ingredient464)
  (dispenses dispenser17 ingredient99)
  (dispenses dispenser188 ingredient301)
  (clean shaker497)
  (clean shot414)
  (clean shot365)
  (empty shaker497)
  (empty shot414)
  (empty shot365)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient464)
  (cocktail-part2 cocktail1 ingredient301)
)
 (:goal
  (and
      (contains shot414 cocktail1)
)))
