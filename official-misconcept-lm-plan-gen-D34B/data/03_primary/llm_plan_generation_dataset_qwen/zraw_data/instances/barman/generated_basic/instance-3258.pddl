(define (problem prob)
 (:domain barman)
 (:objects 
      shaker292 - shaker
      left right - hand
      shot349 shot74 shot303 - shot
      ingredient242 ingredient142 ingredient184 ingredient473 - ingredient
      cocktail259 - cocktail
      dispenser423 dispenser21 dispenser133 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker292)
  (ontable shot349)
  (ontable shot74)
  (ontable shot303)
  (dispenses dispenser423 ingredient242)
  (dispenses dispenser21 ingredient142)
  (dispenses dispenser133 ingredient184)
  (dispenses dispenser9 ingredient473)
  (clean shaker292)
  (clean shot349)
  (clean shot74)
  (clean shot303)
  (empty shaker292)
  (empty shot349)
  (empty shot74)
  (empty shot303)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker292 l0)
  (shaker-level shaker292 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail259 ingredient142)
  (cocktail-part2 cocktail259 ingredient242)
)
 (:goal
  (and
      (contains shot349 cocktail259)
      (contains shot74 cocktail259)
)))
