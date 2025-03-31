(define (problem prob)
 (:domain barman)
 (:objects 
      shaker11 - shaker
      left right - hand
      shot384 shot41 shot444 - shot
      ingredient494 ingredient292 ingredient491 ingredient210 - ingredient
      cocktail1 - cocktail
      dispenser475 dispenser5 dispenser318 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker11)
  (ontable shot384)
  (ontable shot41)
  (ontable shot444)
  (dispenses dispenser475 ingredient494)
  (dispenses dispenser5 ingredient292)
  (dispenses dispenser318 ingredient491)
  (dispenses dispenser21 ingredient210)
  (clean shaker11)
  (clean shot384)
  (clean shot41)
  (clean shot444)
  (empty shaker11)
  (empty shot384)
  (empty shot41)
  (empty shot444)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker11 l0)
  (shaker-level shaker11 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient210)
  (cocktail-part2 cocktail1 ingredient494)
)
 (:goal
  (and
      (contains shot384 cocktail1)
      (contains shot41 cocktail1)
)))
