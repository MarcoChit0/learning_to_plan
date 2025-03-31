(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot347 - shot
      ingredient266 ingredient497 ingredient141 - ingredient
      cocktail1 - cocktail
      dispenser182 dispenser272 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot347)
  (dispenses dispenser182 ingredient266)
  (dispenses dispenser272 ingredient497)
  (dispenses dispenser200 ingredient141)
  (clean shaker177)
  (clean shot347)
  (empty shaker177)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient497)
  (cocktail-part2 cocktail1 ingredient266)
)
 (:goal
  (and
      (contains shot347 cocktail1)
)))
