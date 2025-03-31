(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot40 shot289 - shot
      ingredient259 ingredient243 ingredient443 ingredient171 - ingredient
      cocktail1 - cocktail
      dispenser264 dispenser126 dispenser111 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot40)
  (ontable shot289)
  (dispenses dispenser264 ingredient259)
  (dispenses dispenser126 ingredient243)
  (dispenses dispenser111 ingredient443)
  (dispenses dispenser272 ingredient171)
  (clean shaker207)
  (clean shot40)
  (clean shot289)
  (empty shaker207)
  (empty shot40)
  (empty shot289)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient443)
  (cocktail-part2 cocktail1 ingredient259)
)
 (:goal
  (and
      (contains shot40 cocktail1)
)))
