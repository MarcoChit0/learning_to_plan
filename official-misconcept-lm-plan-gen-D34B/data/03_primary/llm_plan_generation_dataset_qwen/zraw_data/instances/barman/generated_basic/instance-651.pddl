(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot161 shot105 - shot
      ingredient268 ingredient112 ingredient11 - ingredient
      cocktail1 - cocktail
      dispenser376 dispenser6 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot161)
  (ontable shot105)
  (dispenses dispenser376 ingredient268)
  (dispenses dispenser6 ingredient112)
  (dispenses dispenser167 ingredient11)
  (clean shaker491)
  (clean shot161)
  (clean shot105)
  (empty shaker491)
  (empty shot161)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient112)
  (cocktail-part2 cocktail1 ingredient11)
)
 (:goal
  (and
      (contains shot161 cocktail1)
)))
